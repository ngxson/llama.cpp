#pragma once

#include "ggml.h"

#include <vector>
#include <array>

enum vision_arch {
    VISION_ARCH_UNKNOWN,
    VISION_ARCH_LLAVA,
};

enum clip_projector_type {
    CLIP_PROJECTOR_TYPE_UNKNOWN,
    CLIP_PROJECTOR_TYPE_MLP,
};

enum mm_patch_merge {
    MM_PATCH_MERGE_UNKNOWN,
    MM_PATCH_MERGE_FLAT,
    MM_PATCH_MERGE_SPATIAL_UNPAD,
};

struct clip_hparams {
    vision_arch arch = VISION_ARCH_UNKNOWN;

    uint32_t image_size;
    uint32_t patch_size;
    uint32_t hidden_size;
    uint32_t n_intermediate;
    uint32_t projection_dim;
    uint32_t n_head;
    uint32_t n_layer;
    uint32_t max_pos_embd;
    int32_t select_layer = 0;
    bool use_gelu = false;

    float eps;

    clip_projector_type proj_type = CLIP_PROJECTOR_TYPE_UNKNOWN;
    mm_patch_merge mm_patch_merge_type = MM_PATCH_MERGE_FLAT;

    std::array<float, 3> image_mean;
    std::array<float, 3> image_std;

    std::array<int32_t, 32> image_grid_pinpoints;
    int32_t image_crop_resolution;
};

struct clip_layer {
    // attention
    struct ggml_tensor * k_w = NULL;
    struct ggml_tensor * k_b = NULL;
    struct ggml_tensor * q_w = NULL;
    struct ggml_tensor * q_b = NULL;
    struct ggml_tensor * v_w = NULL;
    struct ggml_tensor * v_b = NULL;

    struct ggml_tensor * output_w = NULL;
    struct ggml_tensor * output_b = NULL;

    // layernorm 1
    struct ggml_tensor * norm_in_w = NULL;
    struct ggml_tensor * norm_in_b = NULL;

    // ff
    struct ggml_tensor * ffn_up_w = NULL;
    struct ggml_tensor * ffn_up_b = NULL;

    struct ggml_tensor * ffn_down_w = NULL;
    struct ggml_tensor * ffn_down_b = NULL;

    // layernorm 2
    struct ggml_tensor * norm_out_w = NULL;
    struct ggml_tensor * norm_out_b = NULL;
};

struct clip_vision_model {
    struct clip_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding     = NULL;
    struct ggml_tensor * patch_embeddings    = NULL;
    struct ggml_tensor * patch_bias          = NULL;
    struct ggml_tensor * position_embeddings = NULL;

    struct ggml_tensor * pre_norm_w = NULL;
    struct ggml_tensor * pre_norm_b = NULL;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_norm_w = NULL;
    struct ggml_tensor * post_norm_b = NULL;

    struct ggml_tensor * projection = NULL;

    // LLaVA projection
    struct ggml_tensor * mm_1_w = NULL;
    struct ggml_tensor * mm_1_b = NULL;
    struct ggml_tensor * mm_2_w = NULL;
    struct ggml_tensor * mm_2_b = NULL;

    struct ggml_tensor * image_newline = NULL;
};

struct clip_context {
    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_t sched = nullptr;

    const clip_vision_model * model;

    // temporary output data, to be picked up by llama_decode()
    std::vector<float>     out_embd;  // size == n_tokens * n_embd
    std::vector<llama_pos> out_pos;   // position of each token
};

mm_patch_merge mm_patch_merge_from_name(std::string & name);
clip_projector_type projector_type_from_name(std::string & name);
int clip_n_patches(const clip_context & ctx);
int clip_n_mmproj_embd(const clip_context & ctx);

int32_t llama_encode_vision_internal(clip_context & ctx, llama_batch_img * batch);
void llama_vision_clear_output(clip_context & ctx);
