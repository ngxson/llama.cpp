#include "common.h"
#include "llama.h"

/// demo implementation of llamax library

typedef int32_t llamax_cmpl_id;

// return false to stop generation
// TODO: should we use callback? this will be tricky to maintain
// typedef bool (*llamax_async_callback)(
//     struct llamax_cmpl_response res, void * data);

struct llamax_context;

struct llamax_cmpl_request {
    // one of the 3 types can be used:
    // string, list of tokens or chat messages
    char * content; // for demo, only "content" is handled
    llama_token * tokens;
    size_t n_tokens;
    llama_chat_message * messages;
    size_t n_messages;
    // sampling params
    // TODO: need to support sampling params in plain C
    struct llama_sampling_params sparams;
    // if stream is true, user need a loop to read all partial responses
    bool stream;
    // maybe add stop token, n_predict,... like on server completion endpoint
};
struct llamax_cmpl_request llamax_default_cmpl_request() {
    llama_sampling_params sparams;
    struct llamax_cmpl_request result = {
        NULL,
        NULL,
        0,
        NULL,
        0,
        sparams,
        false,
    };
};

struct llamax_cmpl_response {
    bool is_partial;
    llama_token * tokens;
    size_t n_tokens;
    char * content;
};

// return the ID of this completion
llamax_cmpl_id llamax_create_cmpl(
        struct llamax_context * ctx,
        struct llamax_cmpl_request * req);

// get the response
// return NULL if the completion ID is invalid
struct llamax_cmpl_response * llamax_get_cmpl(
        struct llamax_context * ctx,
        llamax_cmpl_id id);

// free the received response
void llamax_free_response(struct llamax_cmpl_response * res);

// optionally stop the completion mid-way
void llamax_stop_cmpl(
        struct llamax_context * ctx,
        llamax_cmpl_id id);
