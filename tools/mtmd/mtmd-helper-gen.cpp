#include "mtmd.h"
#include "mtmd-helper.h"
#include "mtmd-helper-common.h"
#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef MTMD_INTERNAL_HEADER
#error "mtmd-helper is a public library outside of mtmd. it must not include internal headers"
#endif

//
// Audio generation helpers
//

// maps the 2-letter --tts-lang codes (see tools/tts/README.md) to the language
// names used by the codec_language special tokens
static const std::unordered_map<std::string, std::string> tts_lang_codes = {
    { "cn", "chinese"    },
    { "en", "english"    },
    { "ge", "german"     },
    { "it", "italian"    },
    { "po", "portuguese" },
    { "sp", "spanish"    },
    { "ja", "japanese"   },
    { "ko", "korean"     },
    { "fr", "french"     },
    { "ru", "russian"    },
};

static std::string tts_resolve_lang(const std::string & lang) {
    auto it = tts_lang_codes.find(lang);
    return it != tts_lang_codes.end() ? it->second : lang;
}

static llama_token find_special_token(const llama_vocab * vocab, const std::string & piece) {
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n; t++) {
        if (piece == llama_vocab_get_text(vocab, t)) {
            return t;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static bool write_wav16(std::vector<char> & buf, const std::vector<float> & pcm, int32_t rate) {
    // RIFF chunk sizes are 32-bit; refuse to emit a file with a truncated header
    if (pcm.size() > ((size_t) UINT32_MAX - 36) / 2) {
        return false;
    }
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
    return true;
}

class mtmd_gen_audio_pipeline {
public:
    mtmd_gen_audio_pipeline(llama_context * lctx, mtmd_context * mctx)
        : lctx(lctx), mctx(mctx), model(llama_get_model(lctx)), vocab(llama_model_get_vocab(model)),
          n_embd(llama_model_n_embd(model)), info(mtmd_gen_audio_get_info(mctx)) {}
    virtual ~mtmd_gen_audio_pipeline() = default;

    virtual void reset() = 0;
    virtual int32_t set_input(const mtmd_helper_gen_audio_inp * inp) = 0;
    // sampled may be LLAMA_TOKEN_NULL for pipelines with no discrete backbone token
    // (e.g. continuous/diffusion models); such pipelines read whatever they need
    // directly off h_state_in instead
    virtual int32_t step(llama_token sampled, const float * h_state_in, const float ** h_state_out) = 0;
    virtual int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) = 0;

protected:
    llama_context * lctx;
    mtmd_context  * mctx;
    const llama_model * model;
    const llama_vocab  * vocab;
    int n_embd;
    mtmd_gen_audio_info info;
};

// Qwen3-TTS: dual-track discrete AR (backbone codec_0 + MTP code-predictor for the
// remaining 15 codebooks) into a windowed causal conv/transformer decode (code2wav)
class qwen3tts_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    void reset() override {
        pos = 0;
        codes_buf.clear();
        c2w_state.clear();
        audio_pcm.clear();
        overlay.clear();
        overlay_idx = 0;
        h_state_buf.clear();
        out_buf.clear();
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();

        if (!ensure_cache()) {
            return 1;
        }

        const std::string lang   = tts_resolve_lang((inp->lang && inp->lang[0]) ? inp->lang : "english");
        const llama_token c_lang = find_special_token(vocab, ("<|codec_language_" + lang + "|>").c_str());
        if (c_lang == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: unknown language '%s'\n", lang.c_str());
            return 1;
        }

        std::vector<float> speaker_embd;
        if (inp->speaker_ref) {
            if (!encode_speaker(inp->speaker_ref, speaker_embd)) {
                return 1;
            }
        }

        const int n_e = n_embd;
        auto row = [&](llama_token t) {
            return std::vector<float>(tok_embd.begin() + (size_t) t * n_e,
                                       tok_embd.begin() + (size_t) (t + 1) * n_e);
        };
        auto sum_row = [&](llama_token a, llama_token b) {
            std::vector<float> va = row(a), vb = row(b);
            for (int i = 0; i < n_e; i++) va[(size_t) i] += vb[(size_t) i];
            return va;
        };
        auto sum_vec = [&](llama_token a, const std::vector<float> & vb) {
            std::vector<float> va = row(a);
            for (int i = 0; i < n_e; i++) va[(size_t) i] += vb[(size_t) i];
            return va;
        };

        // upstream chat wrap, then slices: [0:3] role, [3:-5] utterance body
        const std::string full = "<|im_start|>assistant\n" + std::string(inp->prompt, inp->prompt_len) +
                                  "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<llama_token> ids(full.size() + 16);
        int n_ids = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(), ids.data(), (int32_t) ids.size(),
                                   false, true);
        if (n_ids < 8) {
            LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
            return 1;
        }
        ids.resize((size_t) n_ids);

        std::vector<std::vector<float>> prompt;
        for (int i = 0; i < 3; i++) prompt.push_back(row(ids[(size_t) i]));
        prompt.push_back(sum_row(tts_pad, c_think));
        prompt.push_back(sum_row(tts_pad, c_think_b));
        prompt.push_back(sum_row(tts_pad, c_lang));
        prompt.push_back(sum_row(tts_pad, c_think_e));
        if (!speaker_embd.empty()) prompt.push_back(sum_vec(tts_pad, speaker_embd));
        prompt.push_back(sum_row(tts_bos, codec_pad));
        for (int i = 3; i < n_ids - 5; i++) prompt.push_back(sum_row(ids[(size_t) i], codec_pad));
        prompt.push_back(sum_row(tts_eos, codec_pad));
        prompt.push_back(sum_row(tts_pad, codec_bos));

        const int n_prompt = (int) prompt.size();

        // the talker rides the qwen3vl interleaved mrope: positions carry
        // n_pos_per_embd sections laid out [section * n_tokens + i], all
        // equal for a pure text/codec stream
        mrope = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE ||
                llama_model_rope_type(model) == LLAMA_ROPE_TYPE_IMROPE;
        const int n_pos_per_embd = mrope ? 4 : 1;

        std::vector<float> embd_buf((size_t) n_prompt * (size_t) n_e);
        for (int i = 0; i < n_prompt; i++) {
            memcpy(embd_buf.data() + (size_t) i * n_e, prompt[(size_t) i].data(), (size_t) n_e * sizeof(float));
        }

        decode_embd_batch batch_embd(embd_buf.data(), n_prompt, n_pos_per_embd, n_e);
        if (mrope) batch_embd.set_position_mrope_1d(0, 0);
        else       batch_embd.set_position_normal(0, 0);
        batch_embd.batch.logits[n_prompt - 1] = 1;

        if (llama_decode(lctx, batch_embd.batch) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prefill decode failed\n");
            return 1;
        }

        pos = n_prompt;
        top_k = inp->top_k > 0 ? inp->top_k : 50;
        top_p = inp->top_p > 0 ? inp->top_p : 1.0f;
        out_type = inp->out_type;

        // the text stream keeps flowing during generation: the input after
        // frame k adds trailing text row k on top of the codes embedding,
        // then tts_eos, then tts_pad once the utterance is spent
        for (int i = 3; i < n_ids - 5; i++) overlay.push_back(row(ids[(size_t) i]));
        overlay.push_back(row(tts_eos));
        overlay.push_back(row(tts_pad));

        return 0;
    }

    int32_t step(llama_token sampled, const float * h_state_in, const float ** h_state_out) override {
        mtmd_gen_inp inp{};
        inp.type  = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.code0 = sampled - codec_0;
        inp.embd  = const_cast<float *>(h_state_in);
        inp.top_k = top_k;
        inp.top_p = top_p;
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: gen_code process failed\n");
            return 1;
        }

        codes_buf.insert(codes_buf.end(), out.codes, out.codes + out.n_codes);
        if (out.n_codes > 0 && codes_buf.size() / out.n_codes >= window_frames) {
            if (!flush_c2w()) {
                return 1;
            }
        }

        std::vector<float> fb(out.embd, out.embd + n_embd);
        const auto & ov = overlay[std::min(overlay_idx, overlay.size() - 1)];
        for (int i = 0; i < n_embd; i++) fb[(size_t) i] += ov[(size_t) i];
        overlay_idx++;

        const int n_pos_per_embd = mrope ? 4 : 1;
        decode_embd_batch batch_embd(fb.data(), 1, n_pos_per_embd, n_embd);
        if (mrope) batch_embd.set_position_mrope_1d(pos, 0);
        else       batch_embd.set_position_normal(pos, 0);
        batch_embd.batch.logits[0] = 1;
        pos++;

        if (llama_decode(lctx, batch_embd.batch) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return 1;
        }

        const float * he = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(he, he + n_embd);
        *h_state_out = h_state_buf.data();

        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        if (!flush_c2w()) {
            return 1;
        }

        *out_sample_rate = info.sample_rate;
        if (out_n_samples) {
            *out_n_samples = (int64_t) audio_pcm.size();
        }

        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = (const char *) audio_pcm.data();
            *out_data_len = audio_pcm.size() * sizeof(float);
            return 0;
        }

        out_buf.clear();
        if (!write_wav16(out_buf, audio_pcm, info.sample_rate)) {
            LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
            return 1;
        }
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

private:
    bool ensure_cache() {
        if (specials_ok) {
            return true;
        }
        codec_0   = find_special_token(vocab, "<|codec_0|>");
        codec_bos = find_special_token(vocab, "<|codec_bos|>");
        codec_eos = find_special_token(vocab, "<|codec_eos_token|>");
        codec_pad = find_special_token(vocab, "<|codec_pad|>");
        c_think   = find_special_token(vocab, "<|codec_think|>");
        c_think_b = find_special_token(vocab, "<|codec_think_bos|>");
        c_think_e = find_special_token(vocab, "<|codec_think_eos|>");
        tts_pad   = find_special_token(vocab, "<tts_pad>");
        tts_bos   = find_special_token(vocab, "<tts_text_bos>");
        tts_eos   = find_special_token(vocab, "<tts_text_eod>");
        for (llama_token t : { codec_0, codec_bos, codec_eos, codec_pad,
                               c_think, c_think_b, c_think_e,
                               tts_pad, tts_bos, tts_eos }) {
            if (t == LLAMA_TOKEN_NULL) {
                LOG_ERR("mtmd_helper_gen_audio: missing a required special token in vocab\n");
                return false;
            }
        }
        const uint32_t n_tok_embd = llama_model_get_tok_embd(model, nullptr);
        if (n_tok_embd == 0) {
            LOG_ERR("mtmd_helper_gen_audio: model has no token embeddings\n");
            return false;
        }
        tok_embd.resize(n_tok_embd);
        if (llama_model_get_tok_embd(model, tok_embd.data()) != n_tok_embd) {
            LOG_ERR("mtmd_helper_gen_audio: token embedding copy failed\n");
            return false;
        }
        specials_ok = true;
        return true;
    }

    // encodes a reference wav (already loaded as a bitmap) through the mmproj's
    // speaker encoder, returning the single x-vector embedding row it produces
    bool encode_speaker(mtmd_bitmap * bitmap, std::vector<float> & out) {
        if (!mtmd_support_audio(mctx)) {
            LOG_ERR("mtmd_helper_gen_audio: mmproj has no speaker/audio encoder\n");
            return false;
        }
        const std::string  marker = mtmd_default_marker();
        mtmd_input_text     text{ marker.c_str(), marker.size(), false, true };
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bptr = bitmap;
        bool ok = mtmd_tokenize(mctx, chunks, &text, &bptr, 1) == 0;
        if (ok) {
            ok = false;
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
                const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
                if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    continue;
                }
                if (mtmd_encode_chunk(mctx, chunk) != 0) {
                    LOG_ERR("mtmd_helper_gen_audio: speaker encode failed\n");
                    break;
                }
                const float * embd = mtmd_get_output_embd(mctx);
                const size_t  n = (size_t) llama_model_n_embd_inp(model) * mtmd_input_chunk_get_n_tokens(chunk);
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
    bool flush_c2w() {
        if (codes_buf.empty()) {
            return true;
        }
        mtmd_gen_inp inp{};
        inp.type       = MTMD_GEN_PROCESS_TYPE_CODE2WAV;
        inp.codes      = codes_buf.data();
        inp.n_codes    = codes_buf.size();
        inp.state_data = c2w_state.empty() ? nullptr : (const char *) c2w_state.data();
        inp.state_size = c2w_state.size();
        mtmd_gen_out out{};
        if (mtmd_gen_audio_process(mctx, &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: code2wav process failed\n");
            return false;
        }
        audio_pcm.insert(audio_pcm.end(), out.audio, out.audio + out.n_samples);
        c2w_state.assign(out.state_data, out.state_data + out.state_size);
        codes_buf.clear();
        return true;
    }

    // vocab specials fixed across the whole session, looked up once
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


// Chatterbox: single-track discrete AR (the backbone emits the s3 speech tokens
// directly) into a one-shot flow-matching mel decode and NSF-iSTFT vocoder.
// The prompt is pure embedding concat [spkr, cond speech, text, speech bos],
// positions are handled by the backbone (gpt2 wpe / llama learned pos).
class chatterbox_gen_audio_pipeline : public mtmd_gen_audio_pipeline {
public:
    using mtmd_gen_audio_pipeline::mtmd_gen_audio_pipeline;

    void reset() override {
        pos = 0;
        ar_idx = 0;
        codes_buf.clear();
        audio_pcm.clear();
        h_state_buf.clear();
        out_buf.clear();
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) override {
        reset();

        if (!ensure_cache()) {
            return 1;
        }

        if (inp->speaker_ref) {
            // turbo reference chain: loudness normalize the clip, then cap
            // the talker conditioning at 15 s (multilingual: 6 s)
            mtmd_gen_audio_norm_ref(mctx, inp->speaker_ref);
            const size_t t3_cap = (size_t) (t3_cond.empty() ? 15 : 6) * 16000;

            if (!encode_speaker(inp->speaker_ref, spk80)) {
                return 1;
            }
            if (!tokenize_ref(inp->speaker_ref, 10 * 16000, ref_prompt_tokens) ||
                !tokenize_ref(inp->speaker_ref, t3_cap, ref_t3_tokens)) {
                return 1;
            }

            // the tts stage derives the mel-rate prompt features from the
            // same capped reference clip
            {
                const float * pcm = (const float *) mtmd_bitmap_get_data(inp->speaker_ref);
                const size_t  n   = mtmd_bitmap_get_n_bytes(inp->speaker_ref) / sizeof(float);
                ref_pcm16.assign(pcm, pcm + std::min(n, (size_t) 10 * 16000));

                // talker conditioning rows from the voice encoder chain; the
                // multilingual perceiver consumes the embedding rows of the
                // reference speech tokens, built from the fused talker vocab
                std::vector<float> pse;
                if (!t3_cond.empty()) {
                    for (size_t i = 0; i < ref_t3_tokens.size(); i++) {
                        std::vector<float> r(tok_embd.begin() + (size_t) (speech_base + ref_t3_tokens[i]) * n_embd,
                                             tok_embd.begin() + (size_t) (speech_base + ref_t3_tokens[i] + 1) * n_embd);
                        const float * p = speech_pos.data() + i * (size_t) n_embd;
                        for (int j = 0; j < n_embd; j++) {
                            r[(size_t) j] += p[j];
                        }
                        pse.insert(pse.end(), r.begin(), r.end());
                    }
                }
                mtmd_gen_inp gi{};
                gi.type  = MTMD_GEN_PROCESS_TYPE_SPEAKER_COND;
                gi.pcm   = pcm;
                gi.n_pcm = n;
                gi.ref_speech_embd   = pse.empty() ? nullptr : pse.data();
                gi.n_ref_speech_rows = pse.size() / (size_t) n_embd;
                mtmd_gen_out go{};
                if (mtmd_gen_audio_process(mctx, &gi, &go) != 0) {
                    LOG_ERR("mtmd_helper_gen_audio: speaker conditioning failed\n");
                    return 1;
                }
                ref_cond.assign(go.embd, go.embd + go.n_embd);
            }
        }

        const int n_e = n_embd;
        auto row = [&](llama_token t) {
            return std::vector<float>(tok_embd.begin() + (size_t) t * n_e,
                                       tok_embd.begin() + (size_t) (t + 1) * n_e);
        };
        auto add_pos = [&](std::vector<float> & r, const std::vector<float> & tab, int idx) {
            const float * p = tab.data() + (size_t) idx * n_e;
            for (int j = 0; j < n_e; j++) {
                r[(size_t) j] += p[j];
            }
        };
        const bool mtl = !t3_cond.empty();

        std::vector<std::vector<float>> prompt;

        if (mtl) {
            // conditioning: [spkr, perceiver, emotion] block, cloned from the
            // reference clip when present, precomputed default otherwise
            const auto & cond = ref_cond.empty() ? t3_cond : ref_cond;
            for (size_t i = 0; i < cond.size() / (size_t) n_e; i++) {
                prompt.emplace_back(cond.begin() + i * (size_t) n_e, cond.begin() + (i + 1) * (size_t) n_e);
            }
        } else {
            // conditioning: projected speaker row, then the speech token prompt
            // (already fused ids after the text vocab)
            prompt.push_back(ref_cond.empty() ? cond_spkr : ref_cond);
            if (ref_cond.empty()) {
                for (float f : cond_speech_tokens) {
                    prompt.push_back(row(speech_base + (llama_token) f));
                }
            } else {
                for (int32_t t : ref_t3_tokens) {
                    prompt.push_back(row(speech_base + t));
                }
            }
        }

        // text preprocessing per variant: multilingual lowercases with [SPACE]
        // tokens (language tag left to the caller), turbo applies punc_norm
        std::string txt(inp->prompt, inp->prompt_len);
        if (mtl) {
            std::string norm;
            for (char c : txt) {
                if (c == ' ') {
                    norm += "[SPACE]";
                } else if (c >= 'A' && c <= 'Z') {
                    norm += (char) (c - 'A' + 'a');
                } else {
                    norm += c;
                }
            }
            txt = norm;
        } else if (!txt.empty()) {
            if (txt[0] >= 'a' && txt[0] <= 'z') {
                txt[0] = (char) (txt[0] - 'a' + 'A');
            }
            std::string norm;
            for (char c : txt) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    if (!norm.empty() && norm.back() != ' ') {
                        norm += ' ';
                    }
                } else {
                    norm += c;
                }
            }
            auto replace_all = [&norm](const char * from, const char * to) {
                const size_t nf = strlen(from);
                const size_t nt = strlen(to);
                for (size_t p = 0; (p = norm.find(from, p)) != std::string::npos; p += nt) {
                    norm.replace(p, nf, to);
                }
            };
            replace_all("\xE2\x80\xA6", ", "); // ellipsis
            replace_all(":",            ",");
            replace_all("\xE2\x80\x94", "-");  // em dash
            replace_all("\xE2\x80\x93", "-");  // en dash
            replace_all(" ,",           ",");
            replace_all("\xE2\x80\x9C", "\""); // curly double quotes
            replace_all("\xE2\x80\x9D", "\"");
            replace_all("\xE2\x80\x98", "'");  // curly single quotes
            replace_all("\xE2\x80\x99", "'");
            while (!norm.empty() && norm.back() == ' ') {
                norm.pop_back();
            }
            if (!norm.empty() && strchr(".!?-,", norm.back()) == nullptr) {
                norm += '.';
            }
            txt = norm;
        }
        std::vector<llama_token> ids(txt.size() + 16);
        int n_ids = llama_tokenize(vocab, txt.c_str(), (int32_t) txt.size(), ids.data(), (int32_t) ids.size(),
                                   false, true);
        if (n_ids < 1) {
            LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
            return 1;
        }
        ids.resize((size_t) n_ids);
        if (mtl) {
            // only the multilingual prompt wraps the text in start/stop tokens
            ids.insert(ids.begin(), text_start);
            ids.push_back(text_stop);
        }
        for (size_t i = 0; i < ids.size(); i++) {
            prompt.push_back(row(ids[i]));
            if (mtl) {
                add_pos(prompt.back(), text_pos, (int) i);
            }
        }

        // speech bos opens the AR stream
        prompt.push_back(row(llama_vocab_bos(vocab)));
        if (mtl) {
            add_pos(prompt.back(), speech_pos, 0);
        }
        ar_idx = 1;

        const int n_prompt = (int) prompt.size();
        std::vector<float> embd_buf((size_t) n_prompt * (size_t) n_e);
        for (int i = 0; i < n_prompt; i++) {
            memcpy(embd_buf.data() + (size_t) i * n_e, prompt[(size_t) i].data(), (size_t) n_e * sizeof(float));
        }

        decode_embd_batch batch_embd(embd_buf.data(), n_prompt, 1, n_e);
        batch_embd.set_position_normal(0, 0);
        batch_embd.batch.logits[n_prompt - 1] = 1;

        if (llama_decode(lctx, batch_embd.batch) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prefill decode failed\n");
            return 1;
        }

        pos = n_prompt;
        out_type = inp->out_type;
        return 0;
    }

    int32_t step(llama_token sampled, const float * h_state_in, const float ** h_state_out) override {
        GGML_UNUSED(h_state_in);

        // keep only the 6561 s3gen codes, dropping start/stop and oov ids
        if (sampled >= speech_base && sampled - speech_base < 6561) {
            codes_buf.push_back(sampled - speech_base);
        }

        if (!t3_cond.empty()) {
            // multilingual backbone reads embeddings with the learned speech
            // position added on top of the token row
            std::vector<float> e(tok_embd.begin() + (size_t) sampled * n_embd,
                                 tok_embd.begin() + (size_t) (sampled + 1) * n_embd);
            const float * p = speech_pos.data() + (size_t) ar_idx * n_embd;
            for (int j = 0; j < n_embd; j++) {
                e[(size_t) j] += p[j];
            }
            ar_idx++;
            decode_embd_batch batch_embd(e.data(), 1, 1, n_embd);
            batch_embd.set_position_normal(pos, 0);
            batch_embd.batch.logits[0] = 1;
            if (llama_decode(lctx, batch_embd.batch) != 0) {
                LOG_ERR("mtmd_helper_gen_audio: step decode failed\n");
                return 1;
            }
        } else {
            llama_batch batch = llama_batch_get_one(&sampled, 1);
            if (llama_decode(lctx, batch) != 0) {
                LOG_ERR("mtmd_helper_gen_audio: step decode failed\n");
                return 1;
            }
        }
        pos++;

        const float * h = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(h, h + n_embd);
        *h_state_out = h_state_buf.data();
        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) override {
        if (codes_buf.empty()) {
            LOG_ERR("mtmd_helper_gen_audio: no speech tokens generated\n");
            return 1;
        }

        if (t3_cond.empty()) {
            // turbo appends a short silence tail before vocoding
            codes_buf.insert(codes_buf.end(), 3, 4299);
        }

        mtmd_gen_inp gen_inp{};
        gen_inp.type    = MTMD_GEN_PROCESS_TYPE_TTS;
        gen_inp.codes   = codes_buf.data();
        gen_inp.n_codes = codes_buf.size();
        if (!spk80.empty()) {
            gen_inp.ref_spk      = spk80.data();
            gen_inp.ref_tokens   = ref_prompt_tokens.data();
            gen_inp.n_ref_tokens = ref_prompt_tokens.size();
            gen_inp.ref_pcm      = ref_pcm16.data();
            gen_inp.n_ref_pcm    = ref_pcm16.size();
        }
        mtmd_gen_out gen_out{};
        if (mtmd_gen_audio_process(mctx, &gen_inp, &gen_out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: tts decode failed\n");
            return 1;
        }
        audio_pcm.assign(gen_out.audio, gen_out.audio + gen_out.n_samples);

        *out_sample_rate = info.sample_rate;
        *out_n_samples   = (int64_t) audio_pcm.size();
        out_buf.clear();
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV) {
            write_wav16(out_buf, audio_pcm, info.sample_rate);
        } else {
            out_buf.resize(audio_pcm.size() * sizeof(float));
            memcpy(out_buf.data(), audio_pcm.data(), out_buf.size());
        }
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

private:
    bool ensure_cache() {
        if (!tok_embd.empty()) {
            return true;
        }
        // fused vocab layout: [text 0..speech_base) then the speech tokens
        speech_base = find_special_token(vocab, "<|speech_0|>");
        if (speech_base == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: fused speech tokens not found in vocab\n");
            return false;
        }
        n_speech = llama_vocab_n_tokens(vocab) - speech_base;

        // reference config: start_text_token = 255, stop_text_token = 0
        // (only the multilingual prompt wraps the text with them)
        text_start = 255;
        text_stop  = 0;

        const uint32_t n_tok_embd = llama_model_get_tok_embd(model, nullptr);
        if (n_tok_embd != (uint32_t) llama_vocab_n_tokens(vocab) * (uint32_t) n_embd) {
            LOG_ERR("mtmd_helper_gen_audio: unexpected token embedding size\n");
            return false;
        }
        tok_embd.resize(n_tok_embd);
        llama_model_get_tok_embd(model, tok_embd.data());

        // multilingual variant: the mmproj ships a precomputed t3 conditioning
        // block [spkr, perceiver, emotion] and the learned positional tables
        // that the backbone needs added to its input embeddings
        size_t n_t3 = mtmd_gen_audio_read_tensor(mctx, "cond.t3_cond", nullptr, 0);
        if (n_t3 > 0) {
            t3_cond.resize(n_t3);
            if (mtmd_gen_audio_read_tensor(mctx, "cond.t3_cond", t3_cond.data(), n_t3) != n_t3 ||
                n_t3 % (size_t) n_embd != 0) {
                LOG_ERR("mtmd_helper_gen_audio: cond.t3_cond read failed\n");
                return false;
            }
            auto read_table = [&](const char * name, std::vector<float> & dst) {
                size_t n = mtmd_gen_audio_read_tensor(mctx, name, nullptr, 0);
                dst.resize(n);
                if (n == 0 || mtmd_gen_audio_read_tensor(mctx, name, dst.data(), n) != n ||
                    n % (size_t) n_embd != 0) {
                    LOG_ERR("mtmd_helper_gen_audio: %s read failed\n", name);
                    return false;
                }
                return true;
            };
            if (!read_table("t3.text_pos_emb", text_pos) || !read_table("t3.speech_pos_emb", speech_pos)) {
                return false;
            }
            return true;
        }

        cond_spkr.resize((size_t) n_embd);
        if (mtmd_gen_audio_read_tensor(mctx, "cond.spkr_default", cond_spkr.data(), cond_spkr.size()) != (size_t) n_embd) {
            LOG_ERR("mtmd_helper_gen_audio: cond.spkr_default missing\n");
            return false;
        }
        size_t n_ct = mtmd_gen_audio_read_tensor(mctx, "cond.prompt_speech_tokens", nullptr, 0);
        cond_speech_tokens.resize(n_ct);
        if (n_ct == 0 || mtmd_gen_audio_read_tensor(mctx, "cond.prompt_speech_tokens", cond_speech_tokens.data(), n_ct) != n_ct) {
            LOG_ERR("mtmd_helper_gen_audio: cond.prompt_speech_tokens missing\n");
            return false;
        }
        return true;
    }

    // runs the s3 tokenizer on the reference clip, capped to the reference
    // conditioning length, into speech tokens for the flow prompt (10 s cap)
    // and the talker conditioning (6 s cap)
    bool tokenize_ref(mtmd_bitmap * bitmap, size_t n_cap, std::vector<int32_t> & out) {
        const float * pcm = (const float *) mtmd_bitmap_get_data(bitmap);
        const size_t  n   = mtmd_bitmap_get_n_bytes(bitmap) / sizeof(float);

        mtmd_gen_inp gi{};
        gi.type  = MTMD_GEN_PROCESS_TYPE_TOKENIZE;
        gi.pcm   = pcm;
        gi.n_pcm = std::min(n, n_cap);
        mtmd_gen_out go{};
        if (mtmd_gen_audio_process(mctx, &gi, &go) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: reference tokenize failed\n");
            return false;
        }
        out.assign(go.codes, go.codes + go.n_codes);
        return true;
    }

    // runs the speaker encoder on the reference clip through the standard
    // audio chunk path; the CAMPPlus graph outputs the 80-dim s3gen vector
    bool encode_speaker(mtmd_bitmap * bitmap, std::vector<float> & out) {
        if (!mtmd_support_audio(mctx)) {
            LOG_ERR("mtmd_helper_gen_audio: mmproj has no speaker/audio encoder\n");
            return false;
        }
        const std::string  marker = mtmd_default_marker();
        mtmd_input_text     text{ marker.c_str(), marker.size(), false, true };
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bptr = bitmap;
        bool ok = mtmd_tokenize(mctx, chunks, &text, &bptr, 1) == 0;
        if (ok) {
            ok = false;
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
                const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
                if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    continue;
                }
                if (mtmd_encode_chunk(mctx, chunk) != 0) {
                    LOG_ERR("mtmd_helper_gen_audio: speaker encode failed\n");
                    break;
                }
                const float * embd = mtmd_get_output_embd(mctx);
                out.assign(embd, embd + 80);
                ok = true;
                break;
            }
        }
        mtmd_input_chunks_free(chunks);
        return ok;
    }

    std::vector<float> tok_embd;
    std::vector<float> cond_spkr;
    std::vector<float> cond_speech_tokens;
    std::vector<float> t3_cond;
    std::vector<float> text_pos;
    std::vector<float> speech_pos;
    std::vector<float> spk80;
    std::vector<int32_t> ref_prompt_tokens;
    std::vector<int32_t> ref_t3_tokens;
    std::vector<float> ref_pcm16;
    std::vector<float> ref_cond;
    llama_token speech_base = LLAMA_TOKEN_NULL;
    int n_speech = 0;
    llama_token text_start = LLAMA_TOKEN_NULL;
    llama_token text_stop  = LLAMA_TOKEN_NULL;
    int ar_idx = 0;

    llama_pos pos = 0;
    std::vector<int32_t> codes_buf;
    std::vector<float> audio_pcm;
    std::vector<float> h_state_buf;
    std::vector<char> out_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
};

static std::unique_ptr<mtmd_gen_audio_pipeline> make_pipeline(llama_context * lctx, mtmd_context * mctx) {
    switch (mtmd_gen_audio_get_info(mctx).type) {
        case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new qwen3tts_gen_audio_pipeline(lctx, mctx));
        case MTMD_GEN_AUDIO_TYPE_CHATTERBOX:
            return std::unique_ptr<mtmd_gen_audio_pipeline>(new chatterbox_gen_audio_pipeline(lctx, mctx));
        default:
            return nullptr;
    }
}

struct mtmd_helper_gen_audio {
    std::unique_ptr<mtmd_gen_audio_pipeline> pipeline;
};

mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(struct llama_context * lctx, struct mtmd_context * mctx) {
    auto * ctx = new mtmd_helper_gen_audio();
    ctx->pipeline = make_pipeline(lctx, mctx);
    return ctx;
}

void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx) {
    delete ctx;
}

void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * ctx) {
    if (ctx->pipeline) {
        ctx->pipeline->reset();
    }
}

int32_t mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio * ctx, const mtmd_helper_gen_audio_inp * inp) {
    if (!ctx->pipeline) {
        LOG_ERR("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n");
        return 1;
    }
    return ctx->pipeline->set_input(inp);
}

int32_t mtmd_helper_gen_audio_step(mtmd_helper_gen_audio * ctx, llama_token sampled,
                                   const float * h_state_in, const float ** h_state_out) {
    if (!ctx->pipeline) {
        return 1;
    }
    return ctx->pipeline->step(sampled, h_state_in, h_state_out);
}

int32_t mtmd_helper_gen_audio_get_output(mtmd_helper_gen_audio * ctx, int32_t * out_sample_rate,
                                         const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) {
    if (!ctx->pipeline) {
        return 1;
    }
    return ctx->pipeline->get_output(out_sample_rate, out_data, out_data_len, out_n_samples);
}
