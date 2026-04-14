#include "models.h"

#include <vector>

static ggml_tensor * rope_golden_axis(
        ggml_context * ctx0,
        ggml_tensor * cur,   // [n_embd/2]
        ggml_tensor * freqs, // [n_embd/4]
        ggml_tensor * pos    // [n_token]
) {
    auto n_dim = cur->ne[0];
    return ggml_rope_ext(
        ctx0, cur, pos, freqs,
        n_dim, 0, 0,
        1.0f, // freq_base (ignored because we provide freqs directly)
        1.0f, // freq_scale
        0.0f, 1.0f, 0.0f, 0.0f
    );
}

static ggml_tensor * rope_falcon(
    ggml_context * ctx0,
    ggml_tensor * cur,
    ggml_tensor * freqs,
    ggml_tensor * pos,
    float freq_base,
    float freq_scale
) {
    // falcon-ocr style RoPE:
    // - first half of head_dim rotates as normal (1D temporal RoPE)
    // - second half of head_dim rotates with "golden" RoPE (2D RoPE with freq_h and freq_w)
    //   theta = freq_h * pos_h + freq_w * pos_w

    // the tricks for "golden" rope are:
    // - we decompose it into 2 rotations: first rotate by freq_h * pos_h, then rotate by freq_w * pos_w
    // - instead of rotating per-head, we rotate the whole n_embd_half (because "golden" freq different for each head, but ggml_rope only supports one set of freqs broadcasted across all heads)

    const int64_t n_dim  = cur->ne[0];
    const int64_t n_head = cur->ne[1];
    const int64_t n_pos  = cur->ne[2];

    GGML_ASSERT(pos->type == GGML_TYPE_I32);
    GGML_ASSERT(pos->ne[0] == n_pos * 4); // must be m-rope format
    ggml_tensor * pos_t = ggml_view_1d(ctx0, pos, n_pos, 0);
    ggml_tensor * pos_y = ggml_view_1d(ctx0, pos, n_pos, ggml_row_size(pos->type, n_pos));
    ggml_tensor * pos_x = ggml_view_1d(ctx0, pos, n_pos, ggml_row_size(pos->type, n_pos * 2));

    // first half
    ggml_tensor * first;
    {
        first = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            cur->nb[1],
            cur->nb[2],
            0);
        first = ggml_rope_ext(
            ctx0, first, pos_t, nullptr,
            n_dim/2, GGML_ROPE_TYPE_NORMAL,
            0,
            freq_base,
            freq_scale,
            0.0f, 1.0f, 0.0f, 0.0f
        );
    }

    // second half
    ggml_tensor * second;
    {
        const int64_t n_embd_half = n_dim * n_head / 2;
        // printf("shape of cur: %d x %d x %d\n", (int)cur->ne[0], (int)cur->ne[1], (int)cur->ne[2]);
        // printf("shape of freqs: %d x %d x %d\n", (int)freqs->ne[0], (int)freqs->ne[1], (int)freqs->ne[2]);
        // printf("n_embd_half: %d\n", (int)n_embd_half);
        // freqs shape: ne[0]=n_head*n_rot/2, ne[1]=2
        // layout: all h-freqs contiguous first, then all w-freqs
        GGML_ASSERT(freqs->type == GGML_TYPE_F32);
        GGML_ASSERT(freqs->ne[0] == n_embd_half / 2 && freqs->ne[1] == 2);
        // n_embd_half/2 = n_head * n_rot/2 (matches conversion: permute(2,0,1).reshape(2,-1))
        ggml_tensor * freqs_h = ggml_view_1d(ctx0, freqs, n_embd_half / 2, 0);
        ggml_tensor * freqs_w = ggml_view_1d(ctx0, freqs, n_embd_half / 2, ggml_row_size(freqs->type, n_embd_half / 2));

        second = ggml_view_3d(ctx0, cur,
            n_dim/2, n_head, n_pos,
            cur->nb[1],
            cur->nb[2],
            ggml_row_size(cur->type, n_dim/2));

        // flatten head dim; keep n_pos on ne[2] so ggml_rope_ext sees a->ne[2] == pos->ne[0]
        second = ggml_cont(ctx0, second);
        second = ggml_reshape_3d(ctx0, second, n_embd_half, 1, n_pos);

        // apply each axis sequentially
        second = rope_golden_axis(ctx0, second, freqs_w, pos_x);
        second = rope_golden_axis(ctx0, second, freqs_h, pos_y);

        // unflatten head dim
        second = ggml_reshape_3d(ctx0, second, n_dim/2, n_head, n_pos);
    }

    cur = ggml_concat(ctx0, first, second, 0);
    return cur;
}

llm_build_falcon_ocr::llm_build_falcon_ocr(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot * 2);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        // Parameterless RMSNorm (no learned weight)
        cur = ggml_rms_norm(ctx0, inpL, hparams.f_norm_rms_eps);
        cb(cur, "attn_norm", il);

        {
            // note: model doesn't actually use GQA due to "golden" rope enforcing Q dimension
            const int64_t n_head_kv_ratio = 2;
            const int64_t n_head_kv = n_head / n_head_kv_ratio;
            const int64_t n_embd_q = n_embd_head * n_head;
            const int64_t n_embd_k = n_embd_head * n_head_kv;
            const int64_t n_embd_v = n_embd_head * n_head_kv;

            cur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur,
                                    n_embd_head, n_head,    n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], ggml_row_size(cur->type, n_embd_q));
            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur,
                                    n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], ggml_row_size(cur->type, n_embd_k));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur,
                                    n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], ggml_row_size(cur->type, n_embd_v));

            // Parameterless QK-norm (before RoPE)
            Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
            cb(Qcur, "Qcur_normed", il);

            Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
            cb(Kcur, "Kcur_normed", il);

            // repeat K and V to match shape of Q (required by rope_falcon)
            Kcur = ggml_repeat(ctx0, Kcur, Qcur);
            Vcur = ggml_repeat(ctx0, Vcur, Qcur);

            // rope
            Qcur = rope_falcon(ctx0, Qcur, model.layers[il].rope_freqs, inp_pos, freq_base, freq_scale);
            Kcur = rope_falcon(ctx0, Kcur, model.layers[il].rope_freqs, inp_pos, freq_base, freq_scale);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        ggml_tensor * sa_out = ggml_add(ctx0, cur, inpL);
        cb(sa_out, "sa_out", il);

        // Parameterless pre-FFN RMSNorm
        cur = ggml_rms_norm(ctx0, sa_out, hparams.f_norm_rms_eps);
        cb(cur, "ffn_norm", il);

        // Squared ReLU gating FFN
        {
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_RELU_SQR, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, sa_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    // Final RMSNorm (with learned weight)
    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);

    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
