#include "mtmd.h"
#include "mtmd-impl.h"
#include "mtmd-audio.h"
#include "clip-impl.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "src/llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

//
// mtmd_gen_vocoder - base class
//

struct mtmd_gen_vocoder {
    int32_t tok_audio_end = 2048; // end-of-audio token; derived class sets this at init

    virtual ~mtmd_gen_vocoder() = default;

    // reset streaming state at the start of each new audio segment
    virtual void reset() = 0;

    // run one decode step:
    //   backbone_embd      - output embeddings from the backbone (n_embd_backbone floats)
    //   n_embd_backbone    - size of backbone embedding vector
    //   audio_out          - append generated PCM samples here
    //   backbone_embd_out  - fill with the embedding to feed back to the backbone next step
    // returns true when end-of-audio is reached (audio_out may include final flushed samples)
    virtual bool decode(const float        * backbone_embd,
                        int                  n_embd_backbone,
                        std::vector<float> & audio_out,
                        std::vector<float> & backbone_embd_out) = 0;
};

void mtmd_gen_vocoder_free(struct mtmd_gen_vocoder * vocoder) {
    delete vocoder;
}

//
// mtmd_gen_vocoder_lfm2a - LFM2A audio vocoder
//

struct mtmd_gen_vocoder_lfm2a : mtmd_gen_vocoder {
    struct llama_model   * model = nullptr;
    struct llama_context * ctx   = nullptr;
    struct llama_sampler * smpl  = nullptr;

    int n_fft;
    int hop_length;
    std::unique_ptr<mtmd_audio_streaming_istft> istft;

    mtmd_gen_vocoder_lfm2a(int n_fft, int hop_length) : n_fft(n_fft), hop_length(hop_length) {
        istft = std::make_unique<mtmd_audio_streaming_istft>(n_fft, hop_length);
    }

    ~mtmd_gen_vocoder_lfm2a() override {
        if (smpl)  { llama_sampler_free(smpl);  }
        if (ctx)   { llama_free(ctx);           }
        if (model) { llama_model_free(model);   }
    }

    void reset() override {
        istft->reset();
    }

    bool decode(const float        * backbone_embd,
                int                  n_embd_backbone,
                std::vector<float> & audio_out,
                std::vector<float> & backbone_embd_out) override;
};

bool mtmd_gen_vocoder_lfm2a::decode(
        const float        * backbone_embd,
        int                  n_embd_backbone,
        std::vector<float> & audio_out,
        std::vector<float> & backbone_embd_out) {
    const int n_embd_voc = llama_model_n_embd(model);
    if (n_embd_backbone != n_embd_voc) {
        fprintf(stderr, "%s: backbone n_embd (%d) != vocoder n_embd (%d)\n",
                __func__, n_embd_backbone, n_embd_voc);
        return false;
    }

    // run vocoder on backbone embedding (stateless: clear KV cache before each step)
    llama_memory_clear(llama_get_memory(ctx), true);
    {
        struct llama_batch batch = llama_batch_init(1, n_embd_voc, 1);
        batch.n_tokens     = 1;
        std::memcpy(batch.embd, backbone_embd, n_embd_voc * sizeof(float));
        batch.pos[0]       = 0;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = 1;
        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            fprintf(stderr, "%s: vocoder decode failed (%d)\n", __func__, ret);
            return false;
        }
    }

    // sample one audio token from vocoder logits
    const llama_token audio_tok = llama_sampler_sample(smpl, ctx, -1);
    llama_sampler_accept(smpl, audio_tok);

    // end-of-audio: flush ISTFT remaining samples
    if (audio_tok == tok_audio_end) {
        auto flushed = istft->flush();
        audio_out.insert(audio_out.end(), flushed.begin(), flushed.end());
        return true;
    }

    // look up raw token embedding for audio_tok from the vocoder's weight table
    const ggml_tensor * embd_w = llama_model_get_embd_tensor(model);
    GGML_ASSERT(embd_w != nullptr);
    GGML_ASSERT(embd_w->type == GGML_TYPE_F32); // vocoder embedding must be stored as f32
    const int n_embd_tok = (int) embd_w->ne[0];
    std::vector<float> token_embd(n_embd_tok);
    ggml_backend_tensor_get(embd_w, token_embd.data(),
                            (size_t) audio_tok * n_embd_tok * sizeof(float),
                            n_embd_tok * sizeof(float));

    // use token embedding as ISTFT frequency frame input
    {
        const int n_fft_bins = n_fft / 2 + 1;
        const int frame_size = n_fft_bins * 2;
        std::vector<float> frame_spectrum(frame_size, 0.0f);
        std::copy_n(token_embd.data(), std::min(frame_size, n_embd_tok), frame_spectrum.data());
        auto pcm = istft->process_frame(frame_spectrum.data());
        audio_out.insert(audio_out.end(), pcm.begin(), pcm.end());
    }

    // use token embedding as backbone feedback embedding
    backbone_embd_out.resize(n_embd_backbone, 0.0f);
    std::copy_n(token_embd.data(), std::min(n_embd_backbone, n_embd_tok), backbone_embd_out.data());

    return false;
}

//
// factory
//

struct mtmd_gen_vocoder * mtmd_gen_vocoder_init(const char * vocoder_path, struct mtmd_context * mctx) {
    projector_type proj = mtmd_get_projector_type(mctx, /*is_audio=*/true);
    if (proj != PROJECTOR_TYPE_LFM2A) {
        fprintf(stderr, "%s: unsupported audio projector type for vocoder\n", __func__);
        return nullptr;
    }

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model * model = llama_model_load_from_file(vocoder_path, mparams);
    if (!model) {
        fprintf(stderr, "%s: failed to load vocoder model from '%s'\n", __func__, vocoder_path);
        return nullptr;
    }

    int32_t tok_audio_end = 2048;
    int     n_fft         = 1024;
    int     hop_length    = 256;

    char buf[64];
    if (llama_model_meta_val_str(model, "vocoder.n_fft",           buf, sizeof(buf)) > 0) { n_fft         = std::atoi(buf); }
    if (llama_model_meta_val_str(model, "vocoder.hop_length",      buf, sizeof(buf)) > 0) { hop_length    = std::atoi(buf); }
    if (llama_model_meta_val_str(model, "vocoder.audio_end_token", buf, sizeof(buf)) > 0) { tok_audio_end = std::atoi(buf); }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 1;
    cparams.n_batch = 1;
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "%s: failed to create vocoder context\n", __func__);
        llama_model_free(model);
        return nullptr;
    }

    auto sparams = llama_sampler_chain_default_params();
    struct llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(64));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    auto * vocoder        = new mtmd_gen_vocoder_lfm2a(n_fft, hop_length);
    vocoder->model        = model;
    vocoder->ctx          = ctx;
    vocoder->smpl         = smpl;
    vocoder->tok_audio_end = tok_audio_end;

    return vocoder;
}

//
// mtmd_gen_context
//

struct mtmd_gen_context {
    struct llama_context    * lctx    = nullptr;
    struct mtmd_context     * mctx    = nullptr;
    struct mtmd_gen_vocoder * vocoder = nullptr; // not owned

    enum mtmd_gen_state state           = MTMD_GEN_STATE_TEXT;
    llama_token         tok_audio_start = LLAMA_TOKEN_NULL;
    bool                audio_done      = false;

    std::vector<float> audio_samples; // accumulated float PCM
    std::vector<float> next_embd;     // feedback embedding for backbone (filled by decode)
};

struct mtmd_gen_context * mtmd_gen_init(
        struct llama_context    * lctx,
        struct mtmd_context     * mctx,
        struct mtmd_gen_vocoder * vocoder) {
    auto * gen_ctx   = new mtmd_gen_context;
    gen_ctx->lctx    = lctx;
    gen_ctx->mctx    = mctx;
    gen_ctx->vocoder = vocoder;

    // look up the audio-start token from the backbone vocab
    const char * aud_beg = mtmd_get_aud_beg(mctx);
    if (aud_beg) {
        const struct llama_vocab * vocab = llama_model_get_vocab(llama_get_model(lctx));
        llama_token tok = LLAMA_TOKEN_NULL;
        int n = llama_tokenize(vocab, aud_beg, (int32_t) strlen(aud_beg), &tok, 1, false, true);
        if (n == 1) {
            gen_ctx->tok_audio_start = tok;
        } else {
            fprintf(stderr, "%s: warning: aud_beg '%s' did not tokenize to a single token\n",
                    __func__, aud_beg);
        }
    }

    return gen_ctx;
}

void mtmd_gen_free(struct mtmd_gen_context * gen_ctx) {
    delete gen_ctx;
}

enum mtmd_gen_state mtmd_gen_track(struct mtmd_gen_context * gen_ctx, llama_token token) {
    if (gen_ctx->state == MTMD_GEN_STATE_TEXT) {
        if (gen_ctx->tok_audio_start != LLAMA_TOKEN_NULL && token == gen_ctx->tok_audio_start) {
            gen_ctx->state      = MTMD_GEN_STATE_AUDIO;
            gen_ctx->audio_done = false;
            gen_ctx->audio_samples.clear();
            gen_ctx->vocoder->reset();
        }
    } else { // AUDIO
        if (gen_ctx->audio_done) {
            gen_ctx->state = MTMD_GEN_STATE_TEXT;
        }
    }
    return gen_ctx->state;
}

int mtmd_gen_decode(struct mtmd_gen_context * gen_ctx) {
    GGML_ASSERT(gen_ctx->state   == MTMD_GEN_STATE_AUDIO);
    GGML_ASSERT(gen_ctx->vocoder != nullptr);

    auto * lctx   = gen_ctx->lctx;
    const int n_embd = llama_model_n_embd(llama_get_model(lctx));

    const float * backbone_embd = llama_get_embeddings_ith(lctx, -1);
    if (!backbone_embd) {
        fprintf(stderr, "%s: no output embeddings from backbone\n", __func__);
        return -1;
    }

    bool done = gen_ctx->vocoder->decode(backbone_embd, n_embd,
                                          gen_ctx->audio_samples,
                                          gen_ctx->next_embd);
    if (done) {
        gen_ctx->audio_done = true;
        return 0;
    }

    // feed next_embd back to backbone
    const llama_pos next_pos = llama_memory_seq_pos_max(llama_get_memory(lctx), 0) + 1;
    struct llama_batch bb_batch = llama_batch_init(1, n_embd, 1);
    bb_batch.n_tokens     = 1;
    std::memcpy(bb_batch.embd, gen_ctx->next_embd.data(), n_embd * sizeof(float));
    bb_batch.pos[0]       = next_pos;
    bb_batch.n_seq_id[0]  = 1;
    bb_batch.seq_id[0][0] = 0;
    bb_batch.logits[0]    = 1;
    int ret = llama_decode(lctx, bb_batch);
    llama_batch_free(bb_batch);
    if (ret != 0) {
        fprintf(stderr, "%s: backbone decode failed (%d)\n", __func__, ret);
        return ret;
    }

    return 0;
}

const float * mtmd_gen_get_audio(struct mtmd_gen_context * gen_ctx, size_t * n_samples) {
    *n_samples = gen_ctx->audio_samples.size();
    return gen_ctx->audio_samples.data();
}
