#include "models.h"

#include <vector>

llm_build_falcon_ocr::llm_build_falcon_ocr(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_embd_gqa  = hparams.n_embd_v_gqa();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

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
            cur = build_lora_mm(model.layers[il].wqkv, cur);
            cb(cur, "wqkv", il);

            ggml_tensor * Qcur = ggml_view_3d(ctx0, cur,
                                    n_embd_head, n_head, n_tokens, n_embd_head * sizeof(float), cur->nb[1],
                                    0 * sizeof(float) * (n_embd));
            ggml_tensor * Kcur = ggml_view_3d(ctx0, cur,
                                    n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], 1 * sizeof(float) * (n_embd));
            ggml_tensor * Vcur = ggml_view_3d(ctx0, cur,
                                    n_embd_head, n_head_kv, n_tokens, n_embd_head * sizeof(float),
                                    cur->nb[1], 1 * sizeof(float) * (n_embd + n_embd_gqa));

            // Parameterless QK-norm (before RoPE)
            Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
            cb(Qcur, "Qcur_normed", il);

            Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
            cb(Kcur, "Kcur_normed", il);

            // 1D temporal RoPE (first half of head_dim; n_rot = head_dim/2)
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

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
