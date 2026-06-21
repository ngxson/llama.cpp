#include "chat.h"
#include "common.h"
#include "arg.h"
#include "fit.h"

#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

#include <thread>


struct inference_context {
    server_context ctx_server;
    task_params defaults;

    inference_context(const common_params & params) {
        defaults.sampling    = params.sampling;
        defaults.speculative = params.speculative;
        defaults.n_keep      = params.n_keep;
        defaults.n_predict   = 0;
    }

    common_chat_params format_chat(const json & messages) {
        auto meta = ctx_server.get_meta();
        auto & chat_params = meta.chat_params;

        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(messages);
        return common_chat_templates_apply(chat_params.tmpls.get(), inputs);
    }

    void process_prompt(const json & messages) {
        auto chat_params = format_chat(messages);

        server_response_reader rd = ctx_server.get_response_reader();
        {
            server_task task = server_task(SERVER_TASK_TYPE_COMPLETION);
            task.id         = rd.get_new_id();
            task.index      = 0;
            task.params     = defaults;
            task.cli_prompt = chat_params.prompt;
            task.cli        = true;

            const llama_vocab * vocab = llama_model_get_vocab(
                llama_get_model(ctx_server.get_llama_context()));
            for (const auto & token : chat_params.preserved_tokens) {
                auto ids = common_tokenize(vocab, token, false, true);
                if (ids.size() == 1) {
                    task.params.sampling.preserved_tokens.insert(ids[0]);
                }
            }

            rd.post_task({std::move(task)});
        }

        server_task_result_ptr result;
        while (true) {
            result = rd.next(/*should_stop*/ {});
            if (!result) {
                LOG_ERR("Error: no result received\n");
                break;
            }
            if (result->is_error()) {
                LOG_ERR("Error: %s\n", result->to_json().dump().c_str());
                break;
            }
            auto res_final = dynamic_cast<server_task_result_cmpl_final *>(result.get());
            if (res_final) {
                LOG_INF("Processed tokens: %d\n", res_final->timings.prompt_n);
                break;
            }
        }
    }
};

struct cb_context {
    bool enabled = false;

    int n_embd = -1;
    int t_tokens = 0;
    std::vector<std::vector<float>> data; // [layer][embedding]

    void reset() {
        for (auto & layer_data : data) {
            layer_data.clear();
        }
    }

    bool handle_cb(struct ggml_tensor * t, bool ask) {
        std::string name(t->name);
        bool is_l_out = string_starts_with(name, "l_out");
        if (ask) {
            return enabled && is_l_out;
        }
        if (is_l_out) {
            LOG_DBG("Callback: tensor %s, shape: [%lld, %lld, %lld, %lld], type: %d\n",
                t->name, t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->type);

            auto parts = string_split(name, "-");
            GGML_ASSERT(parts.size() == 2);
            int i_layer = std::stoi(parts[1]);
            GGML_ASSERT(i_layer >= 0);

            if (n_embd < 0) {
                n_embd = t->ne[0];
            }
            if (i_layer == 0) {
                t_tokens += t->ne[1];
            }

            if (data.size() <= (size_t)i_layer) {
                data.resize(i_layer + 1);
            }
            size_t add_elem = ggml_nelements(t);
            auto & cur_data = data[i_layer];
            cur_data.resize(cur_data.size() + add_elem);
            ggml_backend_tensor_get(t,
                cur_data.data() + cur_data.size() - add_elem,
                0,
                add_elem * sizeof(float));
            
            if (i_layer == 0) {
                LOG_INF("new data size: %zu (elem), %d (tokens)\n", cur_data.size(), t_tokens);
            }
        }
        return true;
    }

    static bool callback(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * self = (cb_context *) user_data;
        return self->handle_cb(t, ask);
    }
};

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CVECTOR_GENERATOR)) {
        return 1;
    }

    cb_context ctx_cb;
    params.cb_eval = cb_context::callback;
    params.cb_eval_user_data = &ctx_cb;

    params.n_ctx_checkpoints = 0;
    params.warmup = false;

    inference_context ctx(params);

    llama_backend_init();
    llama_numa_init(params.numa);

    if (!ctx.ctx_server.load_model(params)) {
        LOG_ERR("Failed to load the model\n");
        return 1;
    }

    ctx.defaults.sampling = params.sampling;

    std::thread inference_thread([&ctx]() {
        ctx.ctx_server.start_loop();
    });

    json messages = json::array();
    messages.push_back({
        {"role",    "user"},
        {"content", "Hello!"}
    });

    ctx_cb.enabled = true;
    ctx.process_prompt(messages);

    ctx.ctx_server.terminate();
    inference_thread.join();

    llama_backend_free();

    return 0;
}