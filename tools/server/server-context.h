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

struct server_context_impl; // private implementation
struct server_context_impl_deleter {
    void operator()(server_context_impl * p) const;
};

struct server_context {
    std::unique_ptr<server_context_impl, server_context_impl_deleter> impl;

    server_context();
    ~server_context();

    // initialize slots and server-related data
    void init();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(const common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context
    llama_context * get_llama_context() const;
};


struct server_res_generator;

struct server_routes {
    const common_params & params;
    server_context_impl & ctx_server;
    server_http_context & ctx_http; // for reading is_ready
    server_routes(const common_params & params, server_context & ctx_server, server_http_context & ctx_http)
            : params(params), ctx_server(*ctx_server.impl.get()), ctx_http(ctx_http) {
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
