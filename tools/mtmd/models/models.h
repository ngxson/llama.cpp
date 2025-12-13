#pragma once

#include "../clip-graph.h"

struct clip_graph_siglip : clip_graph {
    clip_graph_siglip(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_pixtral : clip_graph {
    clip_graph_pixtral(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen2vl : clip_graph {
    clip_graph_qwen2vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_qwen3vl : clip_graph {
    clip_graph_qwen3vl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_minicpmv : clip_graph {
    clip_graph_minicpmv(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_internvl : clip_graph {
    clip_graph_internvl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llama4 : clip_graph {
    clip_graph_llama4(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_kimivl : clip_graph {
    clip_graph_kimivl(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_cogvlm : clip_graph {
    clip_graph_cogvlm(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_llava : clip_graph {
    clip_graph_llava(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_whisper_enc : clip_graph {
    clip_graph_whisper_enc(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;
};

struct clip_graph_deepseekocr : clip_graph {
    clip_graph_deepseekocr(clip_ctx * ctx, const clip_image_f32 & img) : clip_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * build_sam(ggml_tensor * inp_raw);
    ggml_tensor * build_dsocr_clip(ggml_tensor * patch_embeds);
    ggml_tensor * get_rel_pos(ggml_tensor * rel_pos, ggml_tensor * indices, int q_size, int k_size);
    ggml_tensor * window_partition(ggml_tensor * x, int window);
    ggml_tensor * window_unpartition(ggml_tensor * x, int w, int h, int window);
};
