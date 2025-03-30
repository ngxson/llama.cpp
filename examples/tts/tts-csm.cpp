#include "llama.h"
#include "common.h"
#include "log.h"
#include "arg.h"
#include "mimi-model.h"

#include <vector>
#include <fstream>
#include <float.h>
#include <cstring> // memcpy and strcmp
#include <inttypes.h>

// For more details on how this works, see: https://github.com/ggml-org/llama.cpp/pull/12648

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    By default, model will be downloaded from https://huggingface.co/ggml-org/sesame-csm-1b-GGUF");
    LOG("\n    %s -p \"[0]I have a dream that one day every valley shall be exalted\" -o output.wav", argv[0]);
    LOG("\n");
    LOG("\n    To use a local model, specify the path to the model file:");
    LOG("\n    %s -p ... -m sesame-csm-backbone.gguf -mv kyutai-mimi.gguf -o output.wav", argv[0]);
    LOG("\n");
    LOG("\n    Note: the model need 2 files to run, one ends with '-backbone-<quant>.gguf' and the other ends with '-decoder<quant>.gguf'");
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

    // output_csm_proj is the embeddings output from backbone
    // output_audio_embd is the embeddings output from decoder
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

    params.model         = "sesame-csm-backbone.gguf";
    params.vocoder.model = "kyutai-mimi.gguf";
    params.out_file      = "output.wav";
    params.prompt        = "[0]Hello from Sesame.";
    params.n_predict     = 2048; // CSM's max trained seq length

    // HF model
    params.model_url         = "https://huggingface.co/ggml-org/sesame-csm-1b-GGUF/resolve/main/sesame-csm-backbone.gguf";
    params.vocoder.model_url = "https://huggingface.co/ggml-org/sesame-csm-1b-GGUF/resolve/main/kyutai-mimi.gguf";

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
    params_decoder.n_ctx = 64; // we never use more than this
    string_replace_all(params_decoder.model, "-backbone", "-decoder");
    if (!params_decoder.model_url.empty()) {
        string_replace_all(params_decoder.model_url, "-backbone", "-decoder");
    }

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

    mimi_model mimi(params.vocoder.model.c_str(), true);

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
    llama_batch batch_prompt = llama_batch_init(params.n_batch, 0, 1);
    common_batch_clear(batch_prompt);
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        common_batch_add(batch_prompt, prompt_tokens[i], n_past_bb++, { 0 }, false);
    }
    batch_prompt.logits[batch_prompt.n_tokens - 1] = true;

    // inp_past_embd is the "squashed" embeddings from the decoder
    std::vector<float> inp_past_embd(2048, 0.0f);
    llama_batch batch_past_embd = llama_batch_init(1, inp_past_embd.size(), 1);

    int64_t t_gb_start = ggml_time_ms(); // global start time
    int64_t t_bb       = 0; // backbone time
    int64_t n_bb_gen   = 0; // backbone generation count
    int64_t t_dc       = 0; // decoder time
    int64_t n_dc_gen   = 0; // decoder generation count

    bool is_stop = false;
    std::vector<int> generated_codes;

    // backbone generation loop
    for (int k = 0; k < params.n_predict; ++k) {
        bool is_prompt_processing = k == 0;

        if (!is_prompt_processing) {
            // generate the next RVQ semantic token
            batch_past_embd.n_tokens     = 1;
            batch_past_embd.pos[0]       = n_past_bb++;
            batch_past_embd.seq_id[0][0] = 0;
            batch_past_embd.n_seq_id[0]  = 1;
            batch_past_embd.logits[0]    = true;
            std::memcpy(batch_past_embd.embd, inp_past_embd.data(), inp_past_embd.size() * sizeof(float));
        }

        int64_t t_bb_start = ggml_time_ms();
        if (llama_decode(ctx_bb, is_prompt_processing ? batch_prompt : batch_past_embd) != 0) {
            LOG_ERR("%s: backbone llama_decode() failed\n", __func__);
            return 1;
        }
        n_bb_gen++;
        t_bb += ggml_time_ms() - t_bb_start;

        auto vocab_dc = llama_model_get_vocab(model_dc);
        auto logits   = llama_get_logits_ith(ctx_bb, is_prompt_processing ? (batch_prompt.n_tokens - 1) : 0);
        // for (size_t i = 0; i < 10; ++i) {
        //     printf("%4.2f, ", logits[i]);
        // }
        // printf("\n");

        llama_token semantic_tok = sample_greedy(logits, llama_vocab_n_tokens(vocab_dc));
        printf("%d,", semantic_tok);
        generated_codes.push_back(semantic_tok);

        // for (size_t i = 0; i < 10; ++i) {
        //     printf("%4.2f, ", embd[i]);
        // }
        // printf("\n");


        // decoder generation loop
        inp_past_embd = std::vector<float>(inp_past_embd.size(), 0.0f);
        {
            llama_kv_self_clear(ctx_dc);
            llama_batch batch_embd  = llama_batch_init(1, embd.size(), 1);
            llama_batch batch_token = llama_batch_init(1, 0, 1);

            // first "token" is the latent embeddings from backbone
            {
                batch_embd.n_tokens     = 1;
                batch_embd.pos[0]       = 0;
                batch_embd.seq_id[0][0] = 0;
                batch_embd.n_seq_id[0]  = 1;
                batch_embd.logits[0]    = false;
                std::memcpy(batch_embd.embd, embd.data(), embd.size() * sizeof(float));
            }
            if (llama_decode(ctx_dc, batch_embd) != 0) {
                LOG_ERR("%s: decoder llama_decode(embd) failed\n", __func__);
                return 1;
            }

            // then, decode the semantic_tok to generate acoustic tokens
            llama_token tok = semantic_tok;
            int n_codes = 32;
            int sum_codes = 0; // to check if all codes are 0
            for (int i = 0; i < n_codes; ++i) {
                common_batch_clear(batch_token);
                // encoder vocab is further divided into 32 codebooks, each with 2051 entries
                llama_token inp_tok = tok + 2051*i;
                common_batch_add(batch_token, inp_tok, i+1, { 0 }, true);

                int64_t t_bb_start = ggml_time_ms();
                if (llama_decode(ctx_dc, batch_token) != 0) {
                    LOG_ERR("%s: decoder llama_decode(token) failed\n", __func__);
                    return 1;
                }
                n_dc_gen++;
                t_dc += ggml_time_ms() - t_bb_start;

                // sample the acoustic token
                auto logits = llama_get_logits_ith(ctx_dc, 0);
                llama_token acoustic_tok = sample_greedy(logits, llama_vocab_n_tokens(vocab_dc));

                // discard last code (only for embeddings)
                if (i < n_codes - 1) {
                    printf("%d,", acoustic_tok);
                    tok = acoustic_tok; // next input token
                    sum_codes += acoustic_tok;
                    generated_codes.push_back(acoustic_tok);
                }

                // do progressive hsum of embeddings
                GGML_ASSERT(inp_past_embd.size() == embd.size());
                for (size_t i = 0; i < inp_past_embd.size(); ++i) {
                    inp_past_embd[i] += embd[i];
                }
            }
            printf("\n");

            llama_batch_free(batch_embd);
            llama_batch_free(batch_token);

            // if all codes are 0, then we are done
            is_stop = sum_codes == 0;
        }

        // printf("inp_past_embd, n_past_bb = %d\n", n_past_bb);
        // for (size_t i = 0; i < inp_past_embd.size(); ++i) {
        //     printf("%4.4f, ", inp_past_embd[i]);
        //     if (i == 2) {
        //         printf("... ");
        //         i = inp_past_embd.size() - 4;
        //     }
        // }
        // printf("\n");

        if (is_stop) {
            break;
        }
    }

    // print timing info
    printf("\ntimings:\n");
    printf("  backbone: %" PRId64 " ms, %" PRId64 " generated token (%.2f tok/s)\n", t_bb, n_bb_gen, (float)n_bb_gen*1000/(float)t_bb);
    printf("  decoder:  %" PRId64 " ms, %" PRId64 " generated token (%.2f tok/s)\n", t_dc, n_dc_gen, (float)n_dc_gen*1000/(float)t_dc);
    printf("  total:    %" PRId64 " ms\n\n", ggml_time_ms() - t_gb_start);

    llama_batch_free(batch_prompt);
    llama_batch_free(batch_past_embd);

    printf("decode %zu RVQ tokens into wav...\n", generated_codes.size());
    generated_codes = mimi.transpose_input(generated_codes);
    std::vector<float> wav_data = mimi.decode(generated_codes);

    printf("output wav file: %s\n", params.out_file.c_str());

    if (!save_wav16(params.out_file.c_str(), wav_data, mimi.get_sample_rate())) {
        LOG_ERR("Failed to save wav file\n");
        return 1;
    }

    printf("\n");

    return 0;
}
