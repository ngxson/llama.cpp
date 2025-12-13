#pragma once

#include "ggml.h"
#include "clip-model.h"

#include <cstdint>
#include <vector>
#include <string>

#define MTMD_INTERNAL_HEADER

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
