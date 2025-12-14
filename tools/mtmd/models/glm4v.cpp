#include "models.h"

ggml_cgraph * clip_graph_glm4v::build() {
    GGML_ASSERT(model.patch_bias != nullptr);
    GGML_ASSERT(model.position_embeddings != nullptr);
    GGML_ASSERT(model.class_embedding == nullptr);

    const int batch_size       = 1;
    const int n_pos            = n_patches;
    const int num_position_ids = n_pos * 4; // m-rope requires 4 dim per position

    norm_type norm_t = NORM_TYPE_RMS;

    int mrope_sections[4] = {d_head/4, d_head/4, d_head/4, d_head/4};

    ggml_tensor * inp_raw = build_inp_raw();
    ggml_tensor * inp = ggml_conv_2d(ctx0, model.patch_embeddings_0, inp_raw, patch_size, patch_size, 0, 0, 1, 1);

    GGML_ASSERT(img.nx % (patch_size * 2) == 0);
    GGML_ASSERT(img.ny % (patch_size * 2) == 0);

    // second conv dimension
    {
        auto inp_1 = ggml_conv_2d(ctx0, model.patch_embeddings_1, inp_raw, patch_size, patch_size, 0, 0, 1, 1);
        inp = ggml_add(ctx0, inp, inp_1);

        inp = ggml_permute(ctx0, inp, 1, 2, 0, 3);  // [w, h, c, b] -> [c, w, h, b]
        inp = ggml_cont_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
        inp = ggml_reshape_4d(
            ctx0, inp,
            n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
        inp = ggml_permute(ctx0, inp, 0, 2, 1, 3);
        inp = ggml_cont_3d(
            ctx0, inp,
            n_embd, n_patches_x * n_patches_y, batch_size);
    }

    // add patch bias
    if (model.patch_bias != nullptr) {
        inp = ggml_add(ctx0, inp, model.patch_bias);
        cb(inp, "patch_bias", -1);
    }

    // pre-layernorm (aka post-conv)
    if (model.pre_ln_w) {
        inp = build_norm(inp, model.pre_ln_w, model.pre_ln_b, norm_t, eps, -1);
    }

    // calculate absolute position embedding and apply
    ggml_tensor * learned_pos_embd = resize_position_embeddings();
    learned_pos_embd = ggml_cont_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, n_patches_y, batch_size);
    learned_pos_embd = ggml_reshape_4d(
        ctx0, learned_pos_embd,
        n_embd * 2, n_patches_x / 2, 2, batch_size * (n_patches_y / 2));
    learned_pos_embd = ggml_permute(ctx0, learned_pos_embd, 0, 2, 1, 3);
    learned_pos_embd = ggml_cont_3d(
        ctx0, learned_pos_embd,
        n_embd, n_patches_x * n_patches_y, batch_size);
    inp = ggml_add(ctx0, inp, learned_pos_embd);
    cb(inp, "inp_pos_emb", -1);

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_position_ids);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // build ViT with 2D position embeddings
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        // first half is X axis and second half is Y axis
        return ggml_rope_multi(
                    ctx0, cur, positions, nullptr,
                    d_head/2, mrope_sections, GGML_ROPE_TYPE_VISION,
                    32768, hparams.rope_theta, 1, 0, 1, 32, 1);
    };

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            learned_pos_embd,
                            add_pos);

    cb(cur, "vit_out", -1);

    // post-layernorm
    if (model.post_ln_w) {
        cur = build_norm(cur, model.post_ln_w, model.post_ln_b, norm_t, eps, n_layer);
    }

    // GLM4V projector
    // ref: https://github.com/huggingface/transformers/blob/40dc11cd3eb4126652aa41ef8272525affd4a636/src/transformers/models/glm4v/modeling_glm4v.py#L116-L130

    // patch merger (copied from pixtral)
    {
        int n_merge = hparams.n_merge;
        GGML_ASSERT(n_merge > 0);

        // reshape image tokens to 2D grid
        cur = ggml_reshape_3d(ctx0, cur, n_embd, n_patches_x, n_patches_y);
        cur = ggml_permute(ctx0, cur, 2, 0, 1, 3); // [x, y, n_embd]
        cur = ggml_cont(ctx0, cur);

        // torch.nn.functional.unfold is just an im2col under the hood
        // we just need a dummy kernel to make it work
        ggml_tensor * kernel = ggml_view_3d(ctx0, cur, n_merge, n_merge, cur->ne[2], 0, 0, 0);
        cur = ggml_im2col(ctx0, kernel, cur, n_merge, n_merge, 0, 0, 1, 1, true, inp->type);

        // project to n_embd
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], cur->ne[1] * cur->ne[2]);
        cur = ggml_mul_mat(ctx0, model.mm_patch_merger_w, cur);

        // add bias
        cur = ggml_add(ctx0, cur, model.mm_patch_merger_b);
        cb(cur, "after_patch_merger", -1);
    }

    // FC projector
    {
        cur = ggml_mul_mat(ctx0, model.projection, cur);
        // default LayerNorm (post_projection_norm)
        cur = build_norm(cur, model.mm_post_norm_w, model.mm_post_norm_b, NORM_TYPE_NORMAL, 1e-5, -1);
        cur = ggml_gelu(ctx0, cur);
        cb(cur, "after_fc_proj", -1);
    }

    // FFN projector
    {
        cur = build_ffn(cur,
            model.mm_ffn_up_w, model.mm_ffn_up_b,
            model.mm_ffn_gate_w, model.mm_ffn_gate_b,
            model.mm_ffn_down_w, model.mm_ffn_down_b,
            hparams.ffn_op, -1);
        cb(cur, "after_ffn_proj", -1);
    }

    // build the graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}
