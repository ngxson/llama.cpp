#pragma once

#include "ggml.h"
#include "clip-model.h"

#include <cstdint>
#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

#define WHISPER_ASSERT GGML_ASSERT

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

#define COMMON_SAMPLE_RATE 16000

struct mtmd_audio_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct mtmd_audio_mel {
    int n_len;
    int n_len_org;
    int n_mel;

    std::vector<float> data;
};

struct mtmd_audio_preprocessor {
    const clip_hparams & hparams;

    mtmd_audio_preprocessor(const clip_ctx * ctx): hparams(*clip_get_hparams(ctx)) {}
    
    ~mtmd_audio_preprocessor() = default;
    virtual bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) = 0;
};

struct mtmd_audio_whisper_preprocessor : mtmd_audio_preprocessor {
    mtmd_audio_whisper_preprocessor(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;
};

struct mtmd_audio_whisper_gemma3n : mtmd_audio_preprocessor {
    mtmd_audio_whisper_gemma3n(const clip_ctx * ctx) : mtmd_audio_preprocessor(ctx) {}
    bool preprocess(const float * samples, size_t n_samples, std::vector<mtmd_audio_mel> & output) override;
};
