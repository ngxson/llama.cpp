#include "chat.h"
#include "common.h"
#include "arg.h"
#include "fit.h"

#include "server-common.h"
#include "server-context.h"
#include "server-task.h"

#include "gguf.h"

#include <thread>
#include <fstream>


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

            // never use cache
            task.params.cache_prompt = false;

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

struct activation_data {
    int t_tokens = 0;
    int n_embd   = 0;
    std::vector<std::vector<float>> data; // [layer][embedding]

    // n is the number of tokens to keep; each token spans n_embd floats
    void keep_first_n_tokens(int n) {
        size_t n_elem = (size_t) n * n_embd;
        for (auto & layer_data : data) {
            if (layer_data.size() > n_elem) {
                layer_data.resize(n_elem);
            }
        }
    }
};

struct cb_context {
    bool enabled = false;
    int n_embd = -1;
    activation_data data;

    void reset() {
        data = activation_data();
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
            if (data.n_embd == 0) {
                data.n_embd = t->ne[0];
            }
            if (i_layer == 0) {
                data.t_tokens += t->ne[1];
            }

            if (data.data.size() <= (size_t)i_layer) {
                data.data.resize(i_layer + 1);
            }
            size_t add_elem = ggml_nelements(t);
            auto & cur_data = data.data[i_layer];
            cur_data.resize(cur_data.size() + add_elem);
            ggml_backend_tensor_get(t,
                cur_data.data() + cur_data.size() - add_elem,
                0,
                add_elem * sizeof(float));
            
            if (i_layer == 0) {
                LOG_INF("new data size: %zu (elem), %d (tokens)\n", cur_data.size(), data.t_tokens);
            }
        }
        return true;
    }

    static bool callback(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * self = (cb_context *) user_data;
        return self->handle_cb(t, ask);
    }
};

static std::vector<json> read_jsonl_file(const std::string & path) {
    std::vector<json> messages;
    std::ifstream file(path);
    if (!file.is_open()) {
        GGML_ABORT("Failed to open file: %s\n", path.c_str());
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            messages.push_back(json::parse(line));
        }
    }
    return messages;
}

// returns pair <negative, positive>
static std::vector<std::pair<json, json>> read_pair(const common_params & params) {
    auto pos = read_jsonl_file(params.cvector_positive_file);
    auto neg = read_jsonl_file(params.cvector_negative_file);
    if (pos.size() != neg.size()) {
        LOG_ERR("positive and negative files have different number of lines: %zu vs %zu\n", pos.size(), neg.size());
        GGML_ABORT("Files must have the same number of lines\n");
    }
    std::vector<std::pair<json, json>> pairs;
    for (size_t i = 0; i < pos.size(); ++i) {
        pairs.emplace_back(neg[i], pos[i]);
    }
    return pairs;
}

// write directions as a single 3D tensor [n_embd, n_row, n_layers] in GGUF format.
// each directions[il] is a flat [n_embd, n_row] block (column-major), so
// concatenating layers gives exactly the [n_embd, n_row, n_layers] layout.
static void export_gguf(const std::vector<std::vector<float>> & directions,
        int64_t n_embd, int64_t n_row, int64_t n_layers,
        const std::string & fname) {
    struct ggml_init_params params_ggml = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 1u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx_ggml = ggml_init(params_ggml);

    struct ggml_tensor * t = ggml_new_tensor_3d(ctx_ggml, GGML_TYPE_F32, n_embd, n_row, n_layers);
    ggml_set_name(t, "directions");

    LOG_INF("Exporting directions as tensor [n_embd=%lld, n_row=%lld, n_layers=%lld]\n",
        n_embd, n_row, n_layers);

    std::vector<float> data;
    data.reserve((size_t) n_embd * n_row * n_layers);
    for (int il = 0; il < n_layers; ++il) {
        GGML_ASSERT((int64_t) directions[il].size() == (int64_t) n_embd * n_row);
        data.insert(data.end(), directions[il].begin(), directions[il].end());
    }
    t->data = data.data();

    struct gguf_context * ctx_gguf = gguf_init_empty();
    gguf_set_val_str(ctx_gguf, "general.architecture", "directions");
    gguf_add_tensor(ctx_gguf, t);

    gguf_write_to_file(ctx_gguf, fname.c_str(), false);
    LOG_INF("Saved directions to %s\n", fname.c_str());

    gguf_free(ctx_gguf);
    ggml_free(ctx_ggml);
}

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
    params.cache_prompt = false;
    params.cache_idle_slots = false;

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

    /////////////////////////////////////////

    ctx_cb.enabled = true;
    auto pairs = read_pair(params);

    int64_t n_layers = 0;
    int64_t n_embd   = 0;
    int64_t n_row    = 0;
    std::vector<std::vector<float>> directions;

    for (size_t i = 0; i < pairs.size(); ++i) {
        auto & [neg, pos] = pairs[i];
        LOG_INF("Processing pair (%zu / %zu)\n", i+1, pairs.size());

        ctx_cb.reset();
        ctx.process_prompt(neg);
        activation_data data_neg = std::move(ctx_cb.data);

        ctx_cb.reset();
        ctx.process_prompt(pos);
        activation_data data_pos = std::move(ctx_cb.data);

        int n_tokens_keep = std::min(data_neg.t_tokens, data_pos.t_tokens);
        data_neg.keep_first_n_tokens(n_tokens_keep);
        data_pos.keep_first_n_tokens(n_tokens_keep);

        n_layers = data_neg.data.size();
        n_embd   = ctx_cb.n_embd;
        n_row    = n_row + n_tokens_keep;
        if (directions.empty()) {
            directions.resize(n_layers);
        }

        // calc diff (pos - neg) and append to directions
        for (int il = 0; il < (int) n_layers; ++il) {
            auto & layer_neg = data_neg.data[il];
            auto & layer_pos = data_pos.data[il];
            auto & direction = directions[il];
            size_t n_elem = (size_t) n_embd * n_tokens_keep;
            GGML_ASSERT(layer_neg.size() == n_elem && layer_pos.size() == n_elem);
            size_t prev_size = direction.size();
            direction.resize(prev_size + n_elem);
            for (size_t j = 0; j < n_elem; ++j) {
                direction[prev_size + j] = layer_pos[j] - layer_neg[j];
            }
        }

        GGML_ASSERT((int64_t)directions[0].size() == n_embd * n_row);

        LOG_INF("Saving control vector for pair %zu / %zu, tokens kept: %d, total tokens: %lld\n", i+1, pairs.size(), n_tokens_keep, n_row);
    }

    /////////////////////////////////////////

    std::string out_path = "directions.gguf";
    // write the directions as 3D tensor [n_embd, n_row, n_layers] in GGUF format
    {
        export_gguf(directions, n_embd, n_row, n_layers, out_path);
    }

    /////////////////////////////////////////

    ctx.ctx_server.terminate();
    inference_thread.join();

    llama_backend_free();

    return 0;
}