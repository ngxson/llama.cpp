#include "common.h"
#include "arg.h"
#include "console.h"
#include "log.h"

#include "server-context.h"
#include "server-task.h"

#define PRI(...) LOGV(-1, __VA_ARGS__)

constexpr int POLLING_SECONDS = 1;

static bool g_is_interrupted = false;
static bool should_stop() {
    return g_is_interrupted;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        g_is_interrupted = true;
    }
}
#endif

struct cli_context {
    server_context ctx_server;
    json messages = json::array();

    std::string generate_completion(task_params & params, const json & messages, const std::vector<raw_buffer> & input_files) {
        params.stream = true; // make sure we always use streaming mode
        auto queues = ctx_server.get_queues();
        server_response_reader rd(queues, POLLING_SECONDS);
        {
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id        = queues.first.get_new_id();
            task.params    = params;      // copy
            task.cli_input = messages;    // copy
            task.cli_files = input_files; // copy
            rd.post_task({std::move(task)});
        }

        server_task_result_ptr result = rd.next(should_stop);
        std::string curr_content;
        while (result) {
            if (result->is_error()) {
                PRI("Error: %s\n", result->to_json().dump().c_str());
                return curr_content;
            }
            auto res_partial = dynamic_cast<server_task_result_cmpl_partial *>(result.get());
            if (res_partial) {
                curr_content += res_partial->content;
                PRI("%s", res_partial->content.c_str());
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                break;
            }
            result = rd.next(should_stop);
        }
        return curr_content;
    }
};

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN)) {
        return 1;
    }

    common_init();

    // prefer silent by default; TODO: fix this later
    common_log_set_verbosity_thold(0);

    // struct that contains llama context and inference
    cli_context ctx_cli;

    llama_backend_init();
    llama_numa_init(params.numa);

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (!ctx_cli.ctx_server.load_model(params)) {
        PRI("Failed to load the model\n");
        return 1;
    }
    ctx_cli.ctx_server.init();

    std::thread inference_thread([&ctx_cli]() {
        ctx_cli.ctx_server.start_loop();
    });

    PRI("\n");
    PRI("llama-cli is ready. Type your messages below.\n");
    PRI("\n");

    while (!should_stop()) {
        std::string buffer;
        console::set_display(console::user_input);
        {
            PRI("\n> ");
            std::string line;
            bool another_line = true;
            do {
                another_line = console::readline(line, params.multiline_input);
                buffer += line;
            } while (another_line);
        }
        console::set_display(console::reset);
        PRI("\n");

        if (buffer.empty()) {
            continue;
        }

        try {
            ctx_cli.messages.push_back({
                {"role",    "user"},
                {"content", buffer}
            });
            std::vector<raw_buffer> input_files; // empty for now

            task_params defaults;
            defaults.sampling    = params.sampling;
            defaults.speculative = params.speculative;
            defaults.n_keep      = params.n_keep;
            defaults.n_predict   = params.n_predict;
            defaults.antiprompt  = params.antiprompt;

            std::string assistant_content = ctx_cli.generate_completion(defaults, ctx_cli.messages, input_files);
            ctx_cli.messages.push_back({
                {"role",    "assistant"},
                {"content", assistant_content}
            });
            PRI("\n");
        } catch (const std::exception & ex) {
            PRI("Error: %s\n", ex.what());
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    return 0;
}
