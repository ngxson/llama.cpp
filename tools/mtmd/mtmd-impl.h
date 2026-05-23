#pragma once

#include "mtmd.h"
#include "clip-model.h"

// INTERNAL header, do NOT remove this line under any circumstance
#define MTMD_INTERNAL_HEADER

projector_type mtmd_get_projector_type(const mtmd_context * ctx, bool is_audio);
const clip_hparams * mtmd_get_clip_hparams(const mtmd_context * ctx, bool is_audio);

// Returns the string token that triggers audio output generation (e.g. "<|audio_start|>").
// Returns NULL if this model has no audio generation capability.
const char * mtmd_get_aud_beg(const struct mtmd_context * ctx);
