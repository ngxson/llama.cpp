#include "common.h"
#include "llama.h"
#include "llamax.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <queue>

struct llamax_cmpl {
    std::vector<llama_token> inp_tokens;
    std::queue<struct llamax_cmpl_response *> queue_res;
    struct llama_sampling_params sparams;
};

struct llamax_context {
    struct llama_context * lctx;
    struct llama_model * model;

    std::mutex mutex_tasks;
    llamax_cmpl_id curr_id = 0;
    std::unordered_map<llamax_cmpl_id, llamax_cmpl> tasks;

    void task_delete(llamax_cmpl_id id) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        auto it = tasks.find(id);
        if (it != tasks.end()) {
            tasks.erase(it);
        }
    }

    llamax_cmpl_id task_add(struct llamax_cmpl_request * req) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        llamax_cmpl cmpl;
        // for demo, only "content" is handled
        cmpl.inp_tokens = ::llama_tokenize(lctx, req->content, true, true);
        // TODO: handle other types of input
        cmpl.sparams = req->sparams;
        auto id = curr_id++;
        tasks[id] = cmpl;
    }

    ~llamax_context() {
        // TODO: cancel all tasks
    }
};

llamax_cmpl_id llamax_create_cmpl(
        struct llamax_context * ctx,
        struct llamax_cmpl_request * req) {
    ctx->task_add(req);
}

struct llamax_cmpl_response * llamax_get_cmpl(
        struct llamax_context * ctx,
        llamax_cmpl_id id) {
    
}
