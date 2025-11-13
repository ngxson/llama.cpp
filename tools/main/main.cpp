#include "server-http.h"
#include "server-main.h"

#include "common.h"
#include "arg.h"
#include "log.h"
#include "console.h"

#include <functional>
#include <string>
#include <thread>

// dummy implementation of server_http_context, used for CLI

class server_http_context::Impl {};
server_http_context::server_http_context()
    : pimpl(std::make_unique<server_http_context::Impl>())
{}
server_http_context::~server_http_context() = default;
bool server_http_context::init(const common_params &) { return true; }
bool server_http_context::start() { return true; }
void server_http_context::stop() {}
void server_http_context::get(const std::string &, server_http_context::handler_t) {}

// store the handler globally for using later
server_http_context::handler_t chat_completion_handler = nullptr;
void server_http_context::post(const std::string & path, server_http_context::handler_t handler) {
    if (path == "/chat/completions") {
        chat_completion_handler = handler;
    }
}

static bool should_stop() {
    return false;
}

static void print_response(const std::string & body) {
    auto chunks = string_split(body, "\n");
    for (const auto & c : chunks) {
        if (c.length() < 8) {
            continue;
        }
        if (string_starts_with(c, "data: ")) {
            std::string data = c.substr(6);
            if (data == "[DONE]") {
                return;
            }
            try {
                auto j = json::parse(data);
                if (j.contains("choices") && j["choices"].is_array() && !j["choices"].empty()) {
                    auto & choice = j["choices"][0];
                    if (choice.contains("delta") && choice["delta"].contains("content")) {
                        std::string content = choice["delta"]["content"];
                        LOG("%s", content.c_str());
                    }
                }
            } catch (const std::exception & e) {
                LOG_ERR("Failed to parse JSON chunk: %s\n", e.what());
            }
        }
    }
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN)) {
        return 1;
    }

    common_init();

    std::thread server_thread([params]() {
        common_params params_srv = params;
        params_srv.verbosity = -1; // suppress server logs in CLI mode
        return start_server(params_srv);
    });

    json messages = json::array();

    console::set_display(console::prompt);
    while (server_thread.joinable()) {
        LOG("\n> ");

        // color user input only
        console::set_display(console::user_input);

        std::string line;
        bool another_line = true;
        std::string buffer;
        do {
            another_line = console::readline(line, params.multiline_input);
            buffer += line;
        } while (another_line);

        // done taking input, reset color
        console::set_display(console::reset);

        if (line == "/exit" || line == "/quit") {
            break;
        }

        messages.push_back({
            {"role", "user"},
            {"content", buffer}
        });

        json body;
        body["model"] = "cli-model";
        body["messages"] = messages;
        body["stream"] = true;

        server_http_req req {
            {}, // params
            body.dump(),
            should_stop
        };

        auto res = chat_completion_handler(req);
        if (!res->is_stream()) {
            LOG_ERR("Expected streaming response from server, but got %s\n", res->data.c_str());
            continue;
        }

        while (true) {
            print_response(res->data);
            if (!res->next()) {
                print_response(res->data); // flush last chunk
                break;
            }
        }
    }

    return 0;
}
