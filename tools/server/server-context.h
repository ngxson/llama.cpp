#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <atomic>
#include <cstddef>
#include <cinttypes>
#include <memory>
#include <signal.h>
#include <thread>
#include <unordered_set>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_STARTED, // TODO: this state is only used for setting up the initial prompt processing; maybe merge it with launch_slot_with_task in the future
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

enum server_state {
    SERVER_STATE_LOADING_MODEL,  // Server is starting up, model not fully loaded yet
    SERVER_STATE_READY,          // Server is ready and model is loaded
};

// forward declarations
struct server_slot;

// proxy for std::vector to allow forward declaration of server_slot
struct server_slots_t {
    ~server_slots_t();
    std::vector<server_slot*> data;
    size_t size() const { return data.size(); }
    server_slot & operator[](size_t idx) { return *(data[idx]); }
    server_slot & operator[](size_t idx) const { return *(data[idx]); }
    void clear();
    server_slot & create();
    struct iterator {
        typename std::vector<server_slot*>::iterator it;
        iterator(typename std::vector<server_slot*>::iterator i) : it(i) {}
        server_slot & operator*() { return **it; }
        iterator & operator++() { ++it; return *this; }
        bool operator!=(const iterator& other) const { return it != other.it; }
    };
    iterator begin() { return iterator(data.begin()); }
    iterator end() { return iterator(data.end()); }
};

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init();
    void on_prompt_eval(const server_slot & slot);
    void on_prediction(const server_slot & slot);
    void on_decoded(const server_slots_t & slots);
    void reset_bucket();
};

struct server_context {
public:
    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result llama_init;
    common_init_result llama_init_dft;

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;

    const llama_vocab * vocab = nullptr;
    bool vocab_dft_compatible = true;

    // multimodal
    mtmd_context * mctx = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    common_chat_templates_ptr chat_templates;
    oaicompat_parser_options  oai_parser_opt;

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

private:
    llama_model * model_dft = nullptr;

    llama_context_params cparams_dft;

    llama_batch batch {};

    bool add_bos_token  = true;

    int32_t n_ctx; // total context for all clients / slots

    // slots / clients
    server_slots_t slots;

    int slots_debug = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

public:
    ~server_context();

    // load the model and initialize llama_context
    bool load_model(const common_params & params);

    // initialize slots and server-related data
    void init();

    server_slot * get_slot_by_id(int id);

    server_slot * get_available_slot(const server_task & task);

    void clear_slot(server_slot & slot) const;

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots();

    bool launch_slot_with_task(server_slot & slot, server_task && task);

    bool process_token(completion_token_output & result, server_slot & slot);

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const;

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER);

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0);

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task);

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress);

    void send_final_response(server_slot & slot);

    void send_embedding(const server_slot & slot, const llama_batch & batch);

    void send_rerank(const server_slot & slot, const llama_batch & batch);

    //
    // Functions to process the task
    //

    void process_single_task(server_task && task);

    void update_slots();

    //
    // Utility functions
    //

    int get_slot_n_ctx() const;

    json model_meta() const {
        return json {
            {"vocab_type",  llama_vocab_type       (vocab)},
            {"n_vocab",     llama_vocab_n_tokens   (vocab)},
            {"n_ctx_train", llama_model_n_ctx_train(model)},
            {"n_embd",      llama_model_n_embd     (model)},
            {"n_params",    llama_model_n_params   (model)},
            {"size",        llama_model_size       (model)},
        };
    }
};


struct server_res_generator;

struct server_routes {
    const common_params & params;
    server_context & ctx_server;
    server_http_context & ctx_http; // for reading is_ready
    server_routes(const common_params & params, server_context & ctx_server, server_http_context & ctx_http)
            : params(params), ctx_server(ctx_server), ctx_http(ctx_http) {
        init_routes();
    }

public:
    void init_routes();
    // handlers using lambda function, so that they can capture `this` without `std::bind`
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t get_api_show;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;
private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
                server_task_type type,
                const json & data,
                const std::vector<raw_buffer> & files,
                const std::function<bool()> & should_stop,
                task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
};
