#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

//
// Audio generation helpers
//
// model-specific pipeline logic is dispatched on mtmd_gen_audio_get_info(mctx).type;
// the public surface (init/free/reset/set_input/step/get_output) stays model-agnostic
//

static llama_token find_special_token(const llama_vocab * vocab, const std::string & piece) {
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n; t++) {
        if (piece == llama_vocab_get_text(vocab, t)) {
            return t;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static void write_wav16(std::vector<char> & buf, const std::vector<float> & pcm, int32_t rate) {
    const uint32_t data_sz   = (uint32_t) (pcm.size() * 2);
    const uint32_t riff_sz   = 36 + data_sz;
    const uint32_t fmt_sz    = 16, byte_rate = (uint32_t) rate * 2;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    const uint32_t rate32    = (uint32_t) rate;
    auto put = [&](const void * p, size_t n) {
        const char * c = (const char *) p;
        buf.insert(buf.end(), c, c + n);
    };
    put("RIFF", 4); put(&riff_sz, 4); put("WAVE", 4);
    put("fmt ", 4); put(&fmt_sz, 4);
    put(&fmt, 2); put(&ch, 2); put(&rate32, 4);
    put(&byte_rate, 4); put(&align, 2); put(&bits, 2);
    put("data", 4); put(&data_sz, 4);
    for (float v : pcm) {
        int16_t s = (int16_t) (std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        put(&s, 2);
    }
}

struct mtmd_helper_gen_audio {
    llama_context * lctx;
    mtmd_context  * mctx;
    const llama_model * model;
    const llama_vocab  * vocab;
    int n_embd = 0;
    mtmd_gen_audio_info info;

    // qwen3tts: vocab specials fixed across the whole session, looked up once
    bool specials_ok = false;
    llama_token codec_0    = LLAMA_TOKEN_NULL;
    llama_token codec_bos  = LLAMA_TOKEN_NULL;
    llama_token codec_eos  = LLAMA_TOKEN_NULL;
    llama_token codec_pad  = LLAMA_TOKEN_NULL;
    llama_token c_think    = LLAMA_TOKEN_NULL;
    llama_token c_think_b  = LLAMA_TOKEN_NULL;
    llama_token c_think_e  = LLAMA_TOKEN_NULL;
    llama_token tts_pad    = LLAMA_TOKEN_NULL;
    llama_token tts_bos    = LLAMA_TOKEN_NULL;
    llama_token tts_eos    = LLAMA_TOKEN_NULL;
    std::vector<float> tok_embd; // whole token embedding matrix, n_vocab * n_embd

    // matches hparams.wav_tfm_sliding_window hardcoded in clip.cpp; code2wav
    // batches exactly this many frames per call
    size_t window_frames = 72;

    // per-generation state, cleared by reset()
    bool   mrope = false;
    int    pos   = 0;
    int32_t top_k = 50;
    float   top_p = 1.0f;
    std::vector<int32_t> codes_buf;
    std::vector<uint8_t> c2w_state;
    std::vector<float>   audio_pcm;
    std::vector<std::vector<float>> overlay;
    size_t overlay_idx = 0;
    std::vector<float> h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
};

static bool qwen3tts_ensure_cache(mtmd_helper_gen_audio * ctx) {
    if (ctx->specials_ok) {
        return true;
    }
    ctx->codec_0   = find_special_token(ctx->vocab, "<|codec_0|>");
    ctx->codec_bos = find_special_token(ctx->vocab, "<|codec_bos|>");
    ctx->codec_eos = find_special_token(ctx->vocab, "<|codec_eos_token|>");
    ctx->codec_pad = find_special_token(ctx->vocab, "<|codec_pad|>");
    ctx->c_think   = find_special_token(ctx->vocab, "<|codec_think|>");
    ctx->c_think_b = find_special_token(ctx->vocab, "<|codec_think_bos|>");
    ctx->c_think_e = find_special_token(ctx->vocab, "<|codec_think_eos|>");
    ctx->tts_pad   = find_special_token(ctx->vocab, "<tts_pad>");
    ctx->tts_bos   = find_special_token(ctx->vocab, "<tts_text_bos>");
    ctx->tts_eos   = find_special_token(ctx->vocab, "<tts_text_eod>");
    for (llama_token t : { ctx->codec_0, ctx->codec_bos, ctx->codec_eos, ctx->codec_pad,
                           ctx->c_think, ctx->c_think_b, ctx->c_think_e,
                           ctx->tts_pad, ctx->tts_bos, ctx->tts_eos }) {
        if (t == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: missing a required special token in vocab\n");
            return false;
        }
    }
    const uint32_t n_tok_embd = llama_model_get_tok_embd(ctx->model, nullptr);
    if (n_tok_embd == 0) {
        LOG_ERR("mtmd_helper_gen_audio: model has no token embeddings\n");
        return false;
    }
    ctx->tok_embd.resize(n_tok_embd);
    llama_model_get_tok_embd(ctx->model, ctx->tok_embd.data());
    ctx->specials_ok = true;
    return true;
}

// encodes a reference wav (already loaded as a bitmap) through the mmproj's
// speaker encoder, returning the single x-vector embedding row it produces
static bool qwen3tts_encode_speaker(mtmd_helper_gen_audio * ctx, mtmd_bitmap * bitmap, std::vector<float> & out) {
    if (!mtmd_support_audio(ctx->mctx)) {
        LOG_ERR("mtmd_helper_gen_audio: mmproj has no speaker/audio encoder\n");
        return false;
    }
    const std::string  marker = mtmd_default_marker();
    mtmd_input_text     text{ marker.c_str(), marker.size(), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bptr = bitmap;
    bool ok = mtmd_tokenize(ctx->mctx, chunks, &text, &bptr, 1) == 0;
    if (ok) {
        ok = false;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                continue;
            }
            if (mtmd_encode_chunk(ctx->mctx, chunk) != 0) {
                LOG_ERR("mtmd_helper_gen_audio: speaker encode failed\n");
                break;
            }
            const float * embd = mtmd_get_output_embd(ctx->mctx);
            const size_t  n = (size_t) llama_model_n_embd_inp(ctx->model) * mtmd_input_chunk_get_n_tokens(chunk);
            out.assign(embd, embd + n);
            ok = true;
            break;
        }
    }
    mtmd_input_chunks_free(chunks);
    return ok;
}

// runs one CODE2WAV process() call on whatever is currently buffered, carrying
// the persisted state (KV cache + conv left-context) across batches
static bool qwen3tts_flush_c2w(mtmd_helper_gen_audio * ctx) {
    if (ctx->codes_buf.empty()) {
        return true;
    }
    mtmd_gen_inp inp{};
    inp.type       = MTMD_GEN_PROCESS_TYPE_CODE2WAV;
    inp.codes      = ctx->codes_buf.data();
    inp.n_codes    = ctx->codes_buf.size();
    inp.state_data = ctx->c2w_state.empty() ? nullptr : (const char *) ctx->c2w_state.data();
    inp.state_size = ctx->c2w_state.size();
    mtmd_gen_out out{};
    if (mtmd_gen_audio_process(ctx->mctx, &inp, &out) != 0) {
        LOG_ERR("mtmd_helper_gen_audio: code2wav process failed\n");
        return false;
    }
    ctx->audio_pcm.insert(ctx->audio_pcm.end(), out.audio, out.audio + out.n_samples);
    ctx->c2w_state.assign(out.state_data, out.state_data + out.state_size);
    ctx->codes_buf.clear();
    return true;
}

mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(struct llama_context * lctx, struct mtmd_context * mctx) {
    auto * ctx  = new mtmd_helper_gen_audio();
    ctx->lctx   = lctx;
    ctx->mctx   = mctx;
    ctx->model  = llama_get_model(lctx);
    ctx->vocab  = llama_model_get_vocab(ctx->model);
    ctx->n_embd = llama_model_n_embd(ctx->model);
    ctx->info   = mtmd_gen_audio_get_info(mctx);
    return ctx;
}

void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx) {
    delete ctx;
}

void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * ctx) {
    ctx->pos = 0;
    ctx->codes_buf.clear();
    ctx->c2w_state.clear();
    ctx->audio_pcm.clear();
    ctx->overlay.clear();
    ctx->overlay_idx = 0;
    ctx->h_state_buf.clear();
    ctx->out_buf.clear();
}

int32_t mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio * ctx, const mtmd_helper_gen_audio_inp * inp) {
    mtmd_helper_gen_audio_reset(ctx);

    if (ctx->info.type != MTMD_GEN_AUDIO_TYPE_QWEN3TTS) {
        LOG_ERR("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n");
        return 1;
    }
    if (!qwen3tts_ensure_cache(ctx)) {
        return 1;
    }

    const std::string lang   = inp->lang ? inp->lang : "english";
    const llama_token c_lang = find_special_token(ctx->vocab, ("<|codec_language_" + lang + "|>").c_str());
    if (c_lang == LLAMA_TOKEN_NULL) {
        LOG_ERR("mtmd_helper_gen_audio: unknown language '%s'\n", lang.c_str());
        return 1;
    }

    std::vector<float> speaker_embd;
    if (inp->speaker_ref) {
        if (!qwen3tts_encode_speaker(ctx, inp->speaker_ref, speaker_embd)) {
            return 1;
        }
    }

    const int n_embd = ctx->n_embd;
    auto row = [&](llama_token t) {
        return std::vector<float>(ctx->tok_embd.begin() + (size_t) t * n_embd,
                                   ctx->tok_embd.begin() + (size_t) (t + 1) * n_embd);
    };
    auto sum_row = [&](llama_token a, llama_token b) {
        std::vector<float> va = row(a), vb = row(b);
        for (int i = 0; i < n_embd; i++) va[(size_t) i] += vb[(size_t) i];
        return va;
    };
    auto sum_vec = [&](llama_token a, const std::vector<float> & vb) {
        std::vector<float> va = row(a);
        for (int i = 0; i < n_embd; i++) va[(size_t) i] += vb[(size_t) i];
        return va;
    };

    // upstream chat wrap, then slices: [0:3] role, [3:-5] utterance body
    const std::string full = "<|im_start|>assistant\n" + std::string(inp->prompt, inp->prompt_len) +
                              "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<llama_token> ids(full.size() + 16);
    int n_ids = llama_tokenize(ctx->vocab, full.c_str(), (int32_t) full.size(), ids.data(), (int32_t) ids.size(),
                               false, true);
    if (n_ids < 8) {
        LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
        return 1;
    }
    ids.resize((size_t) n_ids);

    std::vector<std::vector<float>> prompt;
    for (int i = 0; i < 3; i++) prompt.push_back(row(ids[(size_t) i]));
    prompt.push_back(sum_row(ctx->tts_pad, ctx->c_think));
    prompt.push_back(sum_row(ctx->tts_pad, ctx->c_think_b));
    prompt.push_back(sum_row(ctx->tts_pad, c_lang));
    prompt.push_back(sum_row(ctx->tts_pad, ctx->c_think_e));
    if (!speaker_embd.empty()) prompt.push_back(sum_vec(ctx->tts_pad, speaker_embd));
    prompt.push_back(sum_row(ctx->tts_bos, ctx->codec_pad));
    for (int i = 3; i < n_ids - 5; i++) prompt.push_back(sum_row(ids[(size_t) i], ctx->codec_pad));
    prompt.push_back(sum_row(ctx->tts_eos, ctx->codec_pad));
    prompt.push_back(sum_row(ctx->tts_pad, ctx->codec_bos));

    const int n_prompt = (int) prompt.size();

    // the talker rides the qwen3vl interleaved mrope: positions carry
    // n_pos_per_embd sections laid out [section * n_tokens + i], all
    // equal for a pure text/codec stream
    ctx->mrope = llama_model_rope_type(ctx->model) == LLAMA_ROPE_TYPE_MROPE ||
                 llama_model_rope_type(ctx->model) == LLAMA_ROPE_TYPE_IMROPE;
    const int n_pos_per_embd = ctx->mrope ? 4 : 1;

    std::vector<float> embd_buf((size_t) n_prompt * (size_t) n_embd);
    for (int i = 0; i < n_prompt; i++) {
        memcpy(embd_buf.data() + (size_t) i * n_embd, prompt[(size_t) i].data(), (size_t) n_embd * sizeof(float));
    }

    decode_embd_batch batch_embd(embd_buf.data(), n_prompt, n_pos_per_embd, n_embd);
    if (ctx->mrope) batch_embd.set_position_mrope_1d(0, 0);
    else            batch_embd.set_position_normal(0, 0);
    batch_embd.batch.logits[n_prompt - 1] = 1;

    if (llama_decode(ctx->lctx, batch_embd.batch) != 0) {
        LOG_ERR("mtmd_helper_gen_audio: prefill decode failed\n");
        return 1;
    }

    ctx->pos = n_prompt;
    ctx->top_k = inp->top_k > 0 ? inp->top_k : 50;
    ctx->top_p = inp->top_p > 0 ? inp->top_p : 1.0f;
    ctx->out_type = inp->out_type;

    // the text stream keeps flowing during generation: the input after
    // frame k adds trailing text row k on top of the codes embedding,
    // then tts_eos, then tts_pad once the utterance is spent
    for (int i = 3; i < n_ids - 5; i++) ctx->overlay.push_back(row(ids[(size_t) i]));
    ctx->overlay.push_back(row(ctx->tts_eos));
    ctx->overlay.push_back(row(ctx->tts_pad));

    return 0;
}

int32_t mtmd_helper_gen_audio_step(mtmd_helper_gen_audio * ctx, llama_token sampled,
                                   const float * h_state_in, const float ** h_state_out) {
    if (ctx->info.type != MTMD_GEN_AUDIO_TYPE_QWEN3TTS) {
        return 1;
    }

    mtmd_gen_inp inp{};
    inp.type  = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
    inp.code0 = sampled - ctx->codec_0;
    inp.embd  = const_cast<float *>(h_state_in);
    inp.top_k = ctx->top_k;
    inp.top_p = ctx->top_p;
    mtmd_gen_out out{};
    if (mtmd_gen_audio_process(ctx->mctx, &inp, &out) != 0) {
        LOG_ERR("mtmd_helper_gen_audio: gen_code process failed\n");
        return 1;
    }

    ctx->codes_buf.insert(ctx->codes_buf.end(), out.codes, out.codes + out.n_codes);
    if (out.n_codes > 0 && ctx->codes_buf.size() / out.n_codes >= ctx->window_frames) {
        if (!qwen3tts_flush_c2w(ctx)) {
            return 1;
        }
    }

    std::vector<float> fb(out.embd, out.embd + ctx->n_embd);
    const auto & ov = ctx->overlay[std::min(ctx->overlay_idx, ctx->overlay.size() - 1)];
    for (int i = 0; i < ctx->n_embd; i++) fb[(size_t) i] += ov[(size_t) i];
    ctx->overlay_idx++;

    const int n_pos_per_embd = ctx->mrope ? 4 : 1;
    decode_embd_batch batch_embd(fb.data(), 1, n_pos_per_embd, ctx->n_embd);
    if (ctx->mrope) batch_embd.set_position_mrope_1d(ctx->pos, 0);
    else            batch_embd.set_position_normal(ctx->pos, 0);
    batch_embd.batch.logits[0] = 1;
    ctx->pos++;

    if (llama_decode(ctx->lctx, batch_embd.batch) != 0) {
        LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
        return 1;
    }

    const float * he = llama_get_embeddings_ith(ctx->lctx, -1);
    ctx->h_state_buf.assign(he, he + ctx->n_embd);
    *h_state_out = ctx->h_state_buf.data();

    return 0;
}

int32_t mtmd_helper_gen_audio_get_output(mtmd_helper_gen_audio * ctx, int32_t * out_sample_rate,
                                         const char ** out_data, size_t * out_data_len) {
    if (!qwen3tts_flush_c2w(ctx)) {
        return 1;
    }

    *out_sample_rate = ctx->info.sample_rate;

    if (ctx->out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
        *out_data     = (const char *) ctx->audio_pcm.data();
        *out_data_len = ctx->audio_pcm.size() * sizeof(float);
        return 0;
    }

    ctx->out_buf.clear();
    write_wav16(ctx->out_buf, ctx->audio_pcm, ctx->info.sample_rate);
    *out_data     = ctx->out_buf.data();
    *out_data_len = ctx->out_buf.size();
    return 0;
}
