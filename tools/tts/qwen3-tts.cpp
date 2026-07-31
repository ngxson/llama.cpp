// Qwen3-TTS end to end throwaway test driver, using the mtmd_helper_gen_audio_*
// helper (tools/mtmd/mtmd-helper.h) which owns all the Qwen3TTS-specific prompt
// construction, code/audio accumulation and batching. Quick manual test only,
// not polished, will be removed once a real CLI tool lands.

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "common.h"
#include "log.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const char * model_path   = nullptr;
    const char * mmproj_path  = nullptr;
    const char * out_path     = "output.wav";
    const char * speaker_path = nullptr;
    std::string  text;
    std::string  lang    = "english";
    int          max_new = 512;
    int          n_gpu   = 999;

    for (int i = 1; i < argc; i++) {
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", flag); exit(1); }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "-m"))        model_path   = next("-m");
        else if (!strcmp(argv[i], "--mmproj"))  mmproj_path  = next("--mmproj");
        else if (!strcmp(argv[i], "-p"))        text         = next("-p");
        else if (!strcmp(argv[i], "-o"))        out_path     = next("-o");
        else if (!strcmp(argv[i], "--lang"))    lang         = next("--lang");
        else if (!strcmp(argv[i], "--max-new")) max_new      = atoi(next("--max-new"));
        else if (!strcmp(argv[i], "-ngl"))      n_gpu        = atoi(next("-ngl"));
        else if (!strcmp(argv[i], "--speaker")) speaker_path = next("--speaker");
        else {
            fprintf(stderr,
                    "usage: %s -m talker.gguf --mmproj tts.gguf -p \"text\" [-o out.wav] [--lang english] "
                    "[--max-new n] [-ngl n] [--speaker ref.wav]\n", argv[0]);
            return 1;
        }
    }
    if (!model_path || !mmproj_path || text.empty()) {
        fprintf(stderr, "need -m, --mmproj and -p\n");
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { LOG_ERR("failed to load %s\n", model_path); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 4096;
    cparams.n_batch    = 4096;
    cparams.embeddings = true;
    llama_context * lctx = llama_init_from_model(model, cparams);
    if (!lctx) { LOG_ERR("failed to create context\n"); return 1; }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path, model, mtmd_params);
    if (!mctx) { LOG_ERR("failed to load %s\n", mmproj_path); return 1; }
    if (mtmd_gen_audio_get_info(mctx).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        LOG_ERR("mmproj does not support audio generation\n");
        return 1;
    }

    mtmd_helper_bitmap_wrapper speaker_wrapper{ nullptr, nullptr };
    if (speaker_path) {
        speaker_wrapper = mtmd_helper_bitmap_init_from_file(mctx, speaker_path, false);
        if (!speaker_wrapper.bitmap) { LOG_ERR("failed to load %s\n", speaker_path); return 1; }
    }

    mtmd_helper::gen_audio gen(lctx, mctx);
    mtmd_helper_gen_audio_inp inp{};
    inp.prompt      = text.c_str();
    inp.prompt_len  = text.size();
    inp.speaker_ref = speaker_wrapper.bitmap;
    inp.lang        = lang.c_str();
    inp.top_k       = 40;
    inp.top_p       = 0.95f;
    inp.out_type    = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    if (gen.set_input(&inp) != 0) { LOG_ERR("set_input failed\n"); return 1; }
    mtmd_bitmap_free(speaker_wrapper.bitmap);

    // vocab landmarks: only the backbone-level sampling policy (which tokens
    // are valid codec_0 candidates, and which one means EOS) stays here, since
    // that's ordinary LLM sampling, not part of the audio-generation pipeline
    llama_token codec_0_tok = LLAMA_TOKEN_NULL;
    llama_token codec_eos_tok = LLAMA_TOKEN_NULL;
    for (llama_token t = 0; t < llama_vocab_n_tokens(vocab); t++) {
        const char * piece = llama_vocab_get_text(vocab, t);
        if (!strcmp(piece, "<|codec_0|>"))         codec_0_tok   = t;
        else if (!strcmp(piece, "<|codec_eos_token|>")) codec_eos_tok = t;
    }
    if (codec_0_tok == LLAMA_TOKEN_NULL || codec_eos_tok == LLAMA_TOKEN_NULL) {
        LOG_ERR("missing codec special tokens in vocab\n");
        return 1;
    }

    auto sample_codec0 = [&]() -> llama_token {
        const float * logits = llama_get_logits_ith(lctx, -1);
        llama_token   best   = codec_eos_tok;
        float         bestv  = logits[codec_eos_tok];
        for (llama_token t = codec_0_tok; t < codec_0_tok + 2048; t++) {
            if (logits[t] > bestv) { bestv = logits[t]; best = t; }
        }
        return best;
    };

    int n_frames = 0;
    llama_token sampled = sample_codec0();
    const float * h_state = llama_get_embeddings_ith(lctx, -1);
    for (; n_frames < max_new && sampled != codec_eos_tok; n_frames++) {
        const float * h_next = nullptr;
        if (gen.step(sampled, h_state, &h_next) != 0) { LOG_ERR("step failed at frame %d\n", n_frames); return 1; }
        h_state = h_next;
        sampled = sample_codec0();
    }

    int32_t       sample_rate = 0;
    const char *  data        = nullptr;
    size_t        data_len    = 0;
    if (gen.get_output(&sample_rate, &data, &data_len) != 0) { LOG_ERR("get_output failed\n"); return 1; }

    LOG_INF("generated %d frames, %zu bytes of WAV audio (%d Hz)\n", n_frames, data_len, sample_rate);
    FILE * f = fopen(out_path, "wb");
    if (!f) { LOG_ERR("failed to open %s\n", out_path); return 1; }
    fwrite(data, 1, data_len, f);
    fclose(f);
    LOG_INF("wrote %s\n", out_path);

    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
