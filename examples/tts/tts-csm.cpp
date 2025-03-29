#include "llama.h"
#include "common.h"
#include "log.h"
#include "arg.h"

#include <vector>
#include <fstream>
#include <float.h>

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s   TODO    ", argv[0]);
    LOG("\n");
}

// greedy sampling with custom n_vocab
static llama_token sample_greedy(const float * logits, int n_vocab) {
    llama_token max_idx = -1;
    float max_val = -FLT_MAX;
    for (int i = 0; i < n_vocab; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// hook to retrieve the embeddings
static bool ggml_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    std::vector<float> * embd = (std::vector<float> *) user_data;

    if (t && strcmp(t->name, "result_norm") == 0) {
        if (ask) return true;

        auto n_bytes = ggml_nbytes(t);
        embd->resize(n_bytes);
        ggml_backend_tensor_get(t, embd->data(), 0, n_bytes);
        printf("result_norm\n");
        return true;
    }

    return false;
}

int main(int argc, char ** argv) {
    common_params params;

    params.model    = "sesame-csm-backbone.gguf";
    params.out_file = "output.wav";
    params.prompt   = "[0]Hello from Sesame.";

    params.n_predict = 4096;
    params.n_batch   = 8192;
    params.n_ctx     = 8192;

    params.sampling.top_k = 4;
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K, };

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    common_params params_decoder(params); // duplicate the params
    string_replace_all(params_decoder.model, "-backbone", "-decoder");

    std::vector<float> embd;
    params.cb_eval = ggml_callback;
    params.cb_eval_user_data = &embd;
    common_init_result llama_backbone = common_init_from_params(params);
    llama_model   * model_bb = llama_backbone.model.get();
    llama_context * ctx_bb   = llama_backbone.context.get();

    //common_init_result llama_decoder  = common_init_from_params(params_decoder);
    //llama_model   * model_dc = llama_decoder.model.get();
    //llama_context * ctx_dc   = llama_decoder.context.get();

    if (model_bb == nullptr || ctx_bb == nullptr) {
        return ENOENT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_bb);
    llama_tokens prompt_tokens = common_tokenize(vocab, params.prompt, false, true);
    prompt_tokens.insert(prompt_tokens.begin(), llama_vocab_bos(vocab));
    prompt_tokens.insert(prompt_tokens.end(),   llama_vocab_eos(vocab));

    printf("prompt tokens: \n");
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        printf("%d, ", prompt_tokens[i]);
    }
    printf("\n");

    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx_bb, batch) != 0) {
        LOG_ERR("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    //auto vocab_dc = llama_model_get_vocab(model_dc);
    auto logits   = llama_get_logits_ith(ctx_bb, batch.n_tokens - 1);
    //printf("next tok: %d\n", sample_greedy(logits, llama_vocab_n_tokens(vocab_dc)));
    for (size_t i = 0; i < 10; ++i) {
        printf("%4.2f, ", logits[i]);
    }
    printf("next tok: %d\n", sample_greedy(logits, 65632));

    for (size_t i = 0; i < 10; ++i) {
        printf("%4.2f, ", embd[i]);
    }

    return 0;
}
