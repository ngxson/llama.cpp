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

    if (t && (strcmp(t->name, "output_csm_proj") == 0 || strcmp(t->name, "output_audio_embd") == 0)) {
        if (ask) return true;

        embd->resize(ggml_nelements(t));
        ggml_backend_tensor_get(t, embd->data(), 0, ggml_nbytes(t));
        // printf("%s tensor size: %lld, %lld\n", t->name, t->ne[0], t->ne[1]);
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

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    std::vector<float> embd;
    params.cb_eval = ggml_callback;
    params.cb_eval_user_data = &embd;
    params.warmup = false;

    common_params params_decoder(params); // duplicate the params
    string_replace_all(params_decoder.model, "-backbone", "-decoder");

    common_init_result llama_backbone = common_init_from_params(params);
    llama_model   * model_bb = llama_backbone.model.get();
    llama_context * ctx_bb   = llama_backbone.context.get();

    common_init_result llama_decoder  = common_init_from_params(params_decoder);
    llama_model   * model_dc = llama_decoder.model.get();
    llama_context * ctx_dc   = llama_decoder.context.get();

    if (model_bb == nullptr || ctx_bb == nullptr) {
        return ENOENT;
    }

    if (model_dc == nullptr || ctx_dc == nullptr) {
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

    llama_pos n_past_bb = 0;
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    common_batch_clear(batch);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch, prompt_tokens[i], n_past_bb++, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    std::vector<float> inp_past_embd(2048, 0.0f);
    llama_batch batch_past_embd = llama_batch_init(1, inp_past_embd.size(), 1);

    for (int k = 0; k < 4; ++k) {
        if (llama_decode(ctx_bb, k == 0 ? batch : batch_past_embd) != 0) {
            LOG_ERR("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        auto vocab_dc = llama_model_get_vocab(model_dc);
        auto logits   = llama_get_logits_ith(ctx_bb, k == 0 ? (batch.n_tokens - 1) : 0);
        // for (size_t i = 0; i < 10; ++i) {
        //     printf("%4.2f, ", logits[i]);
        // }
        // printf("\n");

        llama_token latent_token = sample_greedy(logits, llama_vocab_n_tokens(vocab_dc));
        // printf("latent_token: %d\n", latent_token);
        printf("%5d, ", latent_token);

        // for (size_t i = 0; i < 10; ++i) {
        //     printf("%4.2f, ", embd[i]);
        // }
        // printf("\n");

        

        // decode
        prompt_tokens.clear();
        prompt_tokens.push_back(latent_token);
        inp_past_embd = std::vector<float>(inp_past_embd.size(), 0.0f);
        {
            llama_kv_self_clear(ctx_dc);
            llama_batch batch_embd  = llama_batch_init(1, embd.size(), 1);
            llama_batch batch_token = llama_batch_init(1, 0, 1);
            {
                batch_embd.n_tokens     = 1;
                batch_embd.pos[0]       = 0;
                batch_embd.seq_id[0][0] = 0;
                batch_embd.n_seq_id[0]  = 1;
                batch_embd.logits[0]    = false;
                memcpy(batch_embd.embd, embd.data(), embd.size() * sizeof(float));
            }
            llama_decode(ctx_dc, batch_embd);
        
            llama_token audio_token = latent_token;
            for (int i = 0; i < 31; ++i) {
                common_batch_clear(batch_token);
                // encoder vocab is further divided into 32 codebooks, each with 2051 entries
                llama_token inp_tok = audio_token + 2051*i;
                common_batch_add(batch_token, inp_tok, i+1, { 0 }, true);
                llama_decode(ctx_dc, batch_token);
                auto logits = llama_get_logits_ith(ctx_dc, 0);
                audio_token = sample_greedy(logits, llama_vocab_n_tokens(vocab_dc));
                printf("%d,", audio_token);
                prompt_tokens.push_back(audio_token);

                GGML_ASSERT(inp_past_embd.size() == embd.size());
                for (size_t i = 0; i < inp_past_embd.size(); ++i) {
                    inp_past_embd[i] += embd[i];
                }
            }
            printf("\n");

            llama_batch_free(batch_embd);
            llama_batch_free(batch_token);
        }

        // prepare for the next iteration
        {
            batch_past_embd.n_tokens     = 1;
            batch_past_embd.pos[0]       = n_past_bb;
            batch_past_embd.seq_id[0][0] = 0;
            batch_past_embd.n_seq_id[0]  = 1;
            batch_past_embd.logits[0]    = true;
            memcpy(batch_past_embd.embd, inp_past_embd.data(), inp_past_embd.size() * sizeof(float));
        }
        n_past_bb++;
    }

    return 0;
}
