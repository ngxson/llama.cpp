#pragma once

#include "utils.hpp"
#include "download.h"

#include <functional>
#include <spawn.h>

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

using router_callback_t = std::function<int(common_params)>;

static void log_server_request(const httplib::Request & req, const httplib::Response & res) {
    // skip GH copilot requests when using default port
    if (req.path == "/v1/health") {
        return;
    }

    // reminder: this function is not covered by httplib's exception handler; if someone does more complicated stuff, think about wrapping it in try-catch

    SRV_INF("request: %s %s %s %d\n", req.method.c_str(), req.path.c_str(), req.remote_addr.c_str(), res.status);

    SRV_DBG("request:  %s\n", req.body.c_str());
    SRV_DBG("response: %s\n", res.body.c_str());
}

static std::unique_ptr<httplib::Server> create_http_server(const common_params & params) {
    std::unique_ptr<httplib::Server> svr;
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_INF("Running with SSL: key = %s, cert = %s\n", params.ssl_file_key.c_str(), params.ssl_file_cert.c_str());
        svr.reset(
            new httplib::SSLServer(params.ssl_file_cert.c_str(), params.ssl_file_key.c_str())
        );
    } else {
        LOG_INF("Running without SSL\n");
        svr.reset(new httplib::Server());
    }
#else
    if (params.ssl_file_key != "" && params.ssl_file_cert != "") {
        LOG_ERR("Server is built without SSL support\n");
        return nullptr;
    }
    svr.reset(new httplib::Server());
#endif

    svr->set_default_headers({{"Server", "llama.cpp"}});
    svr->set_logger(log_server_request);

    svr->set_exception_handler([](const httplib::Request &, httplib::Response & res, const std::exception_ptr & ep) {
        std::string message;
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "Unknown Exception";
        }

        try {
            json formatted_error = format_error_response(message, ERROR_TYPE_SERVER);
            LOG_WRN("got exception: %s\n", formatted_error.dump().c_str());
            res_error(res, formatted_error);
        } catch (const std::exception & e) {
            LOG_ERR("got another exception: %s | while hanlding exception: %s\n", e.what(), message.c_str());
        }
    });

    svr->set_error_handler([](const httplib::Request &, httplib::Response & res) {
        if (res.status == 404) {
            res_error(res, format_error_response("File Not Found", ERROR_TYPE_NOT_FOUND));
        }
        // for other error codes, we skip processing here because it's already done by res_error()
    });

    // set timeouts and change hostname and port
    svr->set_read_timeout (params.timeout_read);
    svr->set_write_timeout(params.timeout_write);

    int n_threads_http = params.n_threads_http;
    if (n_threads_http < 1) {
        // +2 threads for monitoring endpoints
        n_threads_http = std::max(params.n_parallel + 2, (int32_t) std::thread::hardware_concurrency() - 1);
    }
    svr->new_task_queue = [n_threads_http] { return new httplib::ThreadPool(n_threads_http); };

    return svr;
}

struct server_instance {
    pid_t pid;
    int port;
};

namespace router {

std::function<void(int)> shutdown_handler;
std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

inline void signal_handler(int signal) {
    if (router::is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    router::shutdown_handler(signal);
}

// https://gist.github.com/Jacob-Tate/7b326a086cf3f9d46e32315841101109
static std::filesystem::path get_abs_exe_path() {
    #if defined(_MSC_VER)
        wchar_t path[FILENAME_MAX] = { 0 };
        GetModuleFileNameW(nullptr, path, FILENAME_MAX);
        return std::filesystem::path(path);
    #elif defined(__APPLE__) && defined(__MACH__)
        char small_path[PATH_MAX];
        uint32_t size = sizeof(small_path);

        if (_NSGetExecutablePath(small_path, &size) == 0) {
            // resolve any symlinks to get absolute path
            try {
                return std::filesystem::canonical(std::filesystem::path(small_path));
            } catch (...) {
                return std::filesystem::path(small_path);
            }
        } else {
            // buffer was too small, allocate required size and call again
            std::vector<char> buf(size);
            if (_NSGetExecutablePath(buf.data(), &size) == 0) {
                try {
                    return std::filesystem::canonical(std::filesystem::path(buf.data()));
                } catch (...) {
                    return std::filesystem::path(buf.data());
                }
            }
            return std::filesystem::path(std::string(buf.data(), (size > 0) ? size : 0));
        }
    #else
        char path[FILENAME_MAX];
        ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
        return std::filesystem::path(std::string(path, (count > 0) ? count: 0));
    #endif
}

static int create_router_server(common_params params, char ** envp) {
    std::unique_ptr<httplib::Server> svr = create_http_server(params);

    std::mutex m;
    std::map<std::string, server_instance> instances;

    auto add_instance = [&](const std::string & id, server_instance && inst) {
        std::lock_guard<std::mutex> lock(m);
        instances.emplace(id, std::move(inst));
        LOG_INF("added instance id=%s, pid=%d, port=%d\n", id.c_str(), inst.pid, inst.port);
    };

    auto remove_instance = [&](const std::string & id) {
        std::lock_guard<std::mutex> lock(m);
        instances.erase(id);
        LOG_INF("removed instance id=%s\n", id.c_str());
    };

    auto create_instance = [&](const std::string & id, const common_params &) {
        server_instance inst;
        inst.port = rand() % 10000 + 20000; // random port between 20000 and 29999

        pid_t pid = 0;
        {
            // Prepare arguments (pass original or custom ones) using mutable storage for argv
            std::filesystem::path exe_path = get_abs_exe_path();
            std::string path = exe_path.string();

            std::vector<std::string> arg_strs;
            arg_strs.push_back(path);
            arg_strs.push_back("-hf");
            arg_strs.push_back(id);
            arg_strs.push_back("--port");
            arg_strs.push_back(std::to_string(inst.port));

            std::vector<char *> child_argv;
            child_argv.reserve(arg_strs.size() + 1);
            for (auto &s : arg_strs) {
                child_argv.push_back(const_cast<char*>(s.c_str()));
            }
            child_argv.push_back(nullptr);

            LOG_INF("spawning instance %s with hf=%s on port %d\n", path.c_str(), id.c_str(), inst.port);
            if (posix_spawn(&pid, path.c_str(), NULL, NULL, child_argv.data(), envp) != 0) {
                perror("posix_spawn");
                exit(1);
            } else {
                inst.pid = pid;
            }
        }
        add_instance(id, std::move(inst));

        std::thread th([id, pid, &remove_instance]() {
            int status = 0;
            waitpid(pid, &status, 0);
            SRV_INF("instance with pid %d exited with status %d\n", pid, status);
            remove_instance(id);
        });
        if (th.joinable()) {
            th.detach(); // for testing
        } else {
            SRV_ERR("failed to detach thread for instance pid %d\n", inst.pid);
        }
        return 0;
    };

    // just PoC, non-OAI compat
    svr->Get("/models", [instances](const httplib::Request &, httplib::Response & res) {
        auto models = common_list_cached_models();
        json models_json = json::array();
        for (const auto & model : models) {
            models_json.push_back(json {
                {"model",  model.to_string()},
                {"loaded", instances.find(model.to_string()) != instances.end()}, // TODO: non-thread-safe here
            });
        }
        res.set_content(safe_json_to_str(json {{"models", models_json}}), MIMETYPE_JSON);
        res.status = 200;
    });

    svr->Post("/models/load", [&params, &create_instance](const httplib::Request & req, httplib::Response & res) {
        const json body = json::parse(req.body);
        const std::string model_str = json_value(body, "model", std::string());
        if (model_str.empty()) {
            res_error(res, format_error_response("model field is required", ERROR_TYPE_INVALID_REQUEST));
            return;
        }
        if (create_instance(model_str, params) == 0) {
            res.set_content(safe_json_to_str(json {{"status", "loading"}, {"model", model_str}}), MIMETYPE_JSON);
            res.status = 200;
        } else {
            res_error(res, format_error_response("failed to create model instance", ERROR_TYPE_SERVER));
        }
    });

    svr->set_error_handler([&instances](const httplib::Request & req, httplib::Response & res) {
        bool is_unhandled = req.matched_route.empty();
        if (is_unhandled && req.method == "POST") {
            // proxy to the right instance based on HF model id
            const json body = json::parse(req.body);
            const std::string model_str = json_value(body, "model", std::string());
            const auto it = instances.find(model_str);
            if (it != instances.end()) {
                const server_instance & inst = it->second;

                // TODO: support streaming and other methods
                httplib::Client cli("127.0.0.1", inst.port);
                auto cli_res = cli.Post(
                    req.path,
                    req.headers,
                    req.body,
                    MIMETYPE_JSON
                );
                res.status = cli_res->status;
                res.set_content(cli_res->body, cli_res->get_header_value("Content-Type"));
            }
        }
    });

    // run the HTTP server in a thread
    svr->bind_to_port(params.hostname, params.port);
    std::thread t([&]() { svr->listen_after_bind(); });
    svr->wait_until_ready();

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = router::signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (router::signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    router::shutdown_handler = [&](int) {
        svr->stop();
        for (const auto & inst : instances) {
            LOG_INF("terminating instance id=%s, pid=%d\n", inst.first.c_str(), inst.second.pid);
            kill(inst.second.pid, SIGTERM);
        }
    };
    t.join();

    exit(0);
}

} // namespace router
