#include "models.h"

// Chatterbox generation graphs: flow encoder, cfm estimator, s3 tokenizer
// and hift vocoder. Weights come from the source-named tensor map
// (model.cbx_tensors); the speaker encoder lives in chatterbox-spkenc.cpp.

ggml_tensor * cbx_t(const clip_model & model, const std::string & name) {
    auto it = model.cbx_tensors.find(name);
    if (it == model.cbx_tensors.end()) {
        GGML_ABORT("missing chatterbox tensor: %s", name.c_str());
    }
    return it->second;
}

// x [C, T]: y = W x + b with torch Linear weights stored as [in, out]
ggml_tensor * cbx_linear(ggml_context * ctx0, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x) {
    ggml_tensor * y = ggml_mul_mat(ctx0, w, x);
    if (b) {
        y = ggml_add(ctx0, y, b);
    }
    return y;
}

static ggml_tensor * cbx_layer_norm(ggml_context * ctx0, ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, float eps) {
    x = ggml_norm(ctx0, x, eps);
    x = ggml_mul(ctx0, x, w);
    x = ggml_add(ctx0, x, b);
    return x;
}

// x [C, T] -> conv1d over time -> [OC, T_out]; kernel [K, IC, OC], explicit
// host-side asymmetric padding is applied by the caller through pad_l/pad_r
ggml_tensor * cbx_conv1d(ggml_context * ctx0, ggml_tensor * k, ggml_tensor * b, ggml_tensor * x,
                                int stride, int pad_l, int pad_r) {
    ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x)); // [T, C]
    if (pad_l > 0) {
        ggml_tensor * z = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pad_l, xt->ne[1]);
        z = ggml_scale(ctx0, z, 0.0f);
        xt = ggml_concat(ctx0, z, xt, 0);
    }
    if (pad_r > 0) {
        ggml_tensor * z = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, pad_r, xt->ne[1]);
        z = ggml_scale(ctx0, z, 0.0f);
        xt = ggml_concat(ctx0, xt, z, 0);
    }
    ggml_tensor * y = ggml_conv_1d(ctx0, k, xt, stride, 0, 1); // [T_out, OC]
    y = ggml_cont(ctx0, ggml_transpose(ctx0, y));              // [OC, T_out]
    if (b) {
        y = ggml_add(ctx0, y, b);
    }
    return y;
}

// Transformer-XL relative shift: bd [2T-1, T, H] -> [T, T, H] where
// out[j, i, h] = bd[(T-1) - i + j, i, h] (ggml ne0 is the fastest dim).
// Same buffer walk as the espnet rel_shift: left-pad one column, reinterpret
// rows/cols, drop the first row, reinterpret back, keep the first T columns.
static ggml_tensor * cbx_rel_shift(ggml_context * ctx0, ggml_tensor * bd, int T) {
    const int H = (int) bd->ne[2];
    ggml_tensor * z = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, T, H);
    z = ggml_scale(ctx0, z, 0.0f);
    ggml_tensor * p = ggml_cont(ctx0, ggml_concat(ctx0, z, bd, 0));           // [2T, T, H]
    p = ggml_reshape_3d(ctx0, p, T, 2 * T, H);                                // [T, 2T, H]
    p = ggml_view_3d(ctx0, p, T, 2 * T - 1, H, p->nb[1], p->nb[2], p->nb[1]); // drop first row
    p = ggml_cont(ctx0, p);
    p = ggml_reshape_3d(ctx0, p, 2 * T - 1, T, H);                            // [2T-1, T, H]
    p = ggml_view_3d(ctx0, p, T, T, H, p->nb[1], p->nb[2], 0);                // first T columns
    return ggml_cont(ctx0, p);
}

// espnet rel-pos self attention block, pre-norm, x [512, T], pos [512, 2T-1]
static ggml_tensor * cbx_enc_layer(const clip_model & model, ggml_context * ctx0, ggml_tensor * x, ggml_tensor * pos,
                                   const std::string & p, int T) {
    const int n_head = 8;
    const int d_head = 64;
    const float scale = 1.0f / sqrtf((float) d_head);

    ggml_tensor * res = x;
    ggml_tensor * cur = cbx_layer_norm(ctx0, cbx_t(model, p + ".norm_mha.weight"), cbx_t(model, p + ".norm_mha.bias"), x, 1e-5f);

    ggml_tensor * q = cbx_linear(ctx0, cbx_t(model, p + ".self_attn.linear_q.weight"), cbx_t(model, p + ".self_attn.linear_q.bias"), cur);
    ggml_tensor * k = cbx_linear(ctx0, cbx_t(model, p + ".self_attn.linear_k.weight"), cbx_t(model, p + ".self_attn.linear_k.bias"), cur);
    ggml_tensor * v = cbx_linear(ctx0, cbx_t(model, p + ".self_attn.linear_v.weight"), cbx_t(model, p + ".self_attn.linear_v.bias"), cur);
    ggml_tensor * pe = cbx_linear(ctx0, cbx_t(model, p + ".self_attn.linear_pos.weight"), nullptr, pos); // [512, 2T-1]

    q = ggml_reshape_3d(ctx0, q, d_head, n_head, T);
    k = ggml_reshape_3d(ctx0, k, d_head, n_head, T);
    v = ggml_reshape_3d(ctx0, v, d_head, n_head, T);
    pe = ggml_reshape_3d(ctx0, pe, d_head, n_head, 2 * T - 1);

    ggml_tensor * u = cbx_t(model, p + ".self_attn.pos_bias_u"); // [64, 8]
    ggml_tensor * w = cbx_t(model, p + ".self_attn.pos_bias_v");

    ggml_tensor * qu = ggml_add(ctx0, q, ggml_reshape_3d(ctx0, u, d_head, n_head, 1));
    ggml_tensor * qv = ggml_add(ctx0, q, ggml_reshape_3d(ctx0, w, d_head, n_head, 1));

    // per head: [64, T] tensors, scores [T(k), T(q)]
    qu = ggml_cont(ctx0, ggml_permute(ctx0, qu, 0, 2, 1, 3)); // [64, T, 8]
    qv = ggml_cont(ctx0, ggml_permute(ctx0, qv, 0, 2, 1, 3));
    k  = ggml_cont(ctx0, ggml_permute(ctx0, k,  0, 2, 1, 3));
    v  = ggml_cont(ctx0, ggml_permute(ctx0, v,  0, 2, 1, 3));
    pe = ggml_cont(ctx0, ggml_permute(ctx0, pe, 0, 2, 1, 3)); // [64, 2T-1, 8]

    ggml_tensor * ac = ggml_mul_mat(ctx0, k, qu);   // [T(k), T(q), 8]
    ggml_tensor * bd = ggml_mul_mat(ctx0, pe, qv);  // [2T-1, T(q), 8]
    bd = cbx_rel_shift(ctx0, bd, T);                // [T(k), T(q), 8]

    ggml_tensor * scores = ggml_scale(ctx0, ggml_add(ctx0, ac, bd), scale);
    ggml_tensor * probs = ggml_soft_max(ctx0, scores);

    ggml_tensor * o = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v)), probs); // [64, T, 8]
    o = ggml_cont(ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3));                                 // [64, 8, T]
    o = ggml_reshape_2d(ctx0, o, n_head * d_head, T);
    o = cbx_linear(ctx0, cbx_t(model, p + ".self_attn.linear_out.weight"), cbx_t(model, p + ".self_attn.linear_out.bias"), o);
    x = ggml_add(ctx0, res, o);

    res = x;
    cur = cbx_layer_norm(ctx0, cbx_t(model, p + ".norm_ff.weight"), cbx_t(model, p + ".norm_ff.bias"), x, 1e-5f);
    cur = cbx_linear(ctx0, cbx_t(model, p + ".feed_forward.w_1.weight"), cbx_t(model, p + ".feed_forward.w_1.bias"), cur);
    cur = ggml_silu(ctx0, cur); // swish
    cur = cbx_linear(ctx0, cbx_t(model, p + ".feed_forward.w_2.weight"), cbx_t(model, p + ".feed_forward.w_2.bias"), cur);
    x = ggml_add(ctx0, res, cur);
    return x;
}


// mish = x * tanh(softplus(x))
static ggml_tensor * cbx_mish(ggml_context * ctx0, ggml_tensor * x) {
    return ggml_mul(ctx0, x, ggml_tanh(ctx0, ggml_softplus(ctx0, x)));
}

// causal block: conv k3 left-padded, layer norm over channels, mish; x [C, T]
static ggml_tensor * cbx_causal_block(const clip_model & model, ggml_context * ctx0, ggml_tensor * x, const std::string & p) {
    ggml_tensor * k = cbx_t(model, p + ".block.0.weight");
    x = cbx_conv1d(ctx0, k, cbx_t(model, p + ".block.0.bias"), x, 1, (int) k->ne[0] - 1, 0);
    x = cbx_layer_norm(ctx0, cbx_t(model, p + ".block.2.weight"), cbx_t(model, p + ".block.2.bias"), x, 1e-5f);
    return cbx_mish(ctx0, x);
}

// resnet block with time conditioning; x [C, T], temb [1024]
static ggml_tensor * cbx_resnet(const clip_model & model, ggml_context * ctx0, ggml_tensor * x, ggml_tensor * temb, const std::string & p) {
    ggml_tensor * h = cbx_causal_block(model, ctx0, x, p + ".block1");
    ggml_tensor * tproj = cbx_linear(ctx0, cbx_t(model, p + ".mlp.1.weight"), cbx_t(model, p + ".mlp.1.bias"), cbx_mish(ctx0, temb));
    h = ggml_add(ctx0, h, tproj); // broadcast [256, 1] over T
    h = cbx_causal_block(model, ctx0, h, p + ".block2");
    ggml_tensor * res = cbx_conv1d(ctx0, cbx_t(model, p + ".res_conv.weight"), cbx_t(model, p + ".res_conv.bias"), x, 1, 0, 0);
    return ggml_add(ctx0, h, res);
}

// diffusers-style transformer block, full attention; x [256, T]
static ggml_tensor * cbx_tfm_block(const clip_model & model, ggml_context * ctx0, ggml_tensor * x, const std::string & p) {
    const int n_head = 8;
    const int d_head = 64;
    const int T = (int) x->ne[1];

    ggml_tensor * res = x;
    ggml_tensor * cur = cbx_layer_norm(ctx0, cbx_t(model, p + ".norm1.weight"), cbx_t(model, p + ".norm1.bias"), x, 1e-5f);
    ggml_tensor * q = ggml_mul_mat(ctx0, cbx_t(model, p + ".attn1.to_q.weight"), cur);
    ggml_tensor * k = ggml_mul_mat(ctx0, cbx_t(model, p + ".attn1.to_k.weight"), cur);
    ggml_tensor * v = ggml_mul_mat(ctx0, cbx_t(model, p + ".attn1.to_v.weight"), cur);
    q = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, q, d_head, n_head, T), 0, 2, 1, 3)); // [64, T, 8]
    k = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, k, d_head, n_head, T), 0, 2, 1, 3));
    v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, d_head, n_head, T), 0, 2, 1, 3));
    ggml_tensor * scores = ggml_scale(ctx0, ggml_mul_mat(ctx0, k, q), 1.0f / sqrtf((float) d_head));
    ggml_tensor * probs = ggml_soft_max(ctx0, scores);
    ggml_tensor * o = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v)), probs); // [64, T, 8]
    o = ggml_cont(ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3));
    o = ggml_reshape_2d(ctx0, o, n_head * d_head, T);
    o = cbx_linear(ctx0, cbx_t(model, p + ".attn1.to_out.0.weight"), cbx_t(model, p + ".attn1.to_out.0.bias"), o);
    x = ggml_add(ctx0, res, o);

    res = x;
    cur = cbx_layer_norm(ctx0, cbx_t(model, p + ".norm3.weight"), cbx_t(model, p + ".norm3.bias"), x, 1e-5f);
    cur = cbx_linear(ctx0, cbx_t(model, p + ".ff.net.0.proj.weight"), cbx_t(model, p + ".ff.net.0.proj.bias"), cur);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = cbx_linear(ctx0, cbx_t(model, p + ".ff.net.2.weight"), cbx_t(model, p + ".ff.net.2.bias"), cur);
    return ggml_add(ctx0, res, cur);
}

// one estimator evaluation; x_noise [80, T], mu [80, T], spks [80],
// cond [80, T], temb [1024]
static ggml_tensor * cbx_estimator(const clip_model & model, ggml_context * ctx0, ggml_tensor * x_noise, ggml_tensor * mu,
                                   ggml_tensor * spks, ggml_tensor * cond, ggml_tensor * temb, int T) {
    // channels live on ne0, time on ne1: pack along ne0
    ggml_tensor * x = ggml_concat(ctx0, x_noise, mu, 0);                   // [160, T]
    ggml_tensor * spks_b = ggml_repeat(ctx0, ggml_reshape_2d(ctx0, spks, 80, 1), ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, T));
    x = ggml_concat(ctx0, x, spks_b, 0);                                   // [240, T]
    x = ggml_concat(ctx0, x, cond, 0);                                     // [320, T]
    // conv layout is [C, T] with channels contiguous per step; conv helpers
    // transpose internally, the pack above must land on the channel dim
    x = ggml_cont(ctx0, x);

    // down
    ggml_tensor * skip;
    x = cbx_resnet(model, ctx0, x, temb, "a.gen.est.down_blocks.0.0");
    for (int j = 0; model.cbx_tensors.count("a.gen.est.down_blocks.0.1." + std::to_string(j) + ".norm1.weight"); j++) {
        x = cbx_tfm_block(model, ctx0, x, "a.gen.est.down_blocks.0.1." + std::to_string(j));
    }
    skip = x;
    {
        ggml_tensor * k = cbx_t(model, "a.gen.est.down_blocks.0.2.weight");
        x = cbx_conv1d(ctx0, k, cbx_t(model, "a.gen.est.down_blocks.0.2.bias"), x, 1, (int) k->ne[0] - 1, 0);
    }

    // mid
    for (int i = 0; model.cbx_tensors.count("a.gen.est.mid_blocks." + std::to_string(i) + ".0.block1.block.0.weight"); i++) {
        const std::string mp = "a.gen.est.mid_blocks." + std::to_string(i);
        x = cbx_resnet(model, ctx0, x, temb, mp + ".0");
        for (int j = 0; model.cbx_tensors.count(mp + ".1." + std::to_string(j) + ".norm1.weight"); j++) {
            x = cbx_tfm_block(model, ctx0, x, mp + ".1." + std::to_string(j));
        }
    }

    // up with skip
    x = ggml_concat(ctx0, x, skip, 0);                                     // [512, T]
    x = cbx_resnet(model, ctx0, x, temb, "a.gen.est.up_blocks.0.0");
    for (int j = 0; model.cbx_tensors.count("a.gen.est.up_blocks.0.1." + std::to_string(j) + ".norm1.weight"); j++) {
        x = cbx_tfm_block(model, ctx0, x, "a.gen.est.up_blocks.0.1." + std::to_string(j));
    }
    {
        ggml_tensor * k = cbx_t(model, "a.gen.est.up_blocks.0.2.weight");
        x = cbx_conv1d(ctx0, k, cbx_t(model, "a.gen.est.up_blocks.0.2.bias"), x, 1, (int) k->ne[0] - 1, 0);
    }

    x = cbx_causal_block(model, ctx0, x, "a.gen.est.final_block");
    x = cbx_conv1d(ctx0, cbx_t(model, "a.gen.est.final_proj.weight"), cbx_t(model, "a.gen.est.final_proj.bias"), x, 1, 0, 0); // [80, T]
    return x;
}

// s3 tokenizer encoder: whisper style log-mel [T, 128] in, two stride 2
// convs to token rate, 6 pre-norm attention blocks with neox rope on q/k and
// an fsmn memory over the value projection, then the fsq down projection.
// output is the post-tanh 8-dim code [8, T / 4], rounded to base 3 tokens on
// the host.
static ggml_cgraph * cbx_build_s3tok(const clip_model & model, ggml_context * ctx0, ggml_cgraph * gf, int T) {
    const int n_head = 20;
    const int d_head = 64;
    const int T1 = (T - 1) / 2 + 1;
    const int T2 = (T1 - 1) / 2 + 1;

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, 128);
    ggml_set_name(inp, "inp_raw");
    ggml_set_input(inp);

    ggml_tensor * pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T2);
    ggml_set_name(pos, "inp_pos");
    ggml_set_input(pos);

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, inp)); // [128, T]
    x = cbx_conv1d(ctx0, cbx_t(model, "a.s3tok.encoder.conv1.weight"), cbx_t(model, "a.s3tok.encoder.conv1.bias"), x, 2, 1, 1);
    x = ggml_gelu_erf(ctx0, x);
    x = cbx_conv1d(ctx0, cbx_t(model, "a.s3tok.encoder.conv2.weight"), cbx_t(model, "a.s3tok.encoder.conv2.bias"), x, 2, 1, 1);
    x = ggml_gelu_erf(ctx0, x); // [1280, T2]

    for (int li = 0; model.cbx_tensors.count("a.s3tok.encoder.blocks." + std::to_string(li) + ".attn_ln.weight"); li++) {
        const std::string p = "a.s3tok.encoder.blocks." + std::to_string(li) + ".attn";

        ggml_tensor * res = x;
        ggml_tensor * cur = cbx_layer_norm(ctx0, cbx_t(model, p + "_ln.weight"), cbx_t(model, p + "_ln.bias"), x, 1e-5f);
        ggml_tensor * q = cbx_linear(ctx0, cbx_t(model, p + ".query.weight"), cbx_t(model, p + ".query.bias"), cur);
        ggml_tensor * k = ggml_mul_mat(ctx0, cbx_t(model, p + ".key.weight"), cur);
        ggml_tensor * v = cbx_linear(ctx0, cbx_t(model, p + ".value.weight"), cbx_t(model, p + ".value.bias"), cur);

        // fsmn memory: depthwise conv k31 over time on the value projection,
        // residual, added to the projected attention context
        ggml_tensor * fsm = ggml_cont(ctx0, ggml_transpose(ctx0, v)); // [T2, 1280]
        {
            ggml_tensor * w = cbx_t(model, p + ".fsmn_block.weight");
            ggml_tensor * m = ggml_conv_1d_dw(ctx0, w, fsm, 1, ((int) w->ne[0] - 1) / 2, 1);
            fsm = ggml_add(ctx0, ggml_reshape_2d(ctx0, m, fsm->ne[0], fsm->ne[1]), fsm);
        }
        fsm = ggml_cont(ctx0, ggml_transpose(ctx0, fsm)); // [1280, T2]

        q = ggml_reshape_3d(ctx0, q, d_head, n_head, T2);
        k = ggml_reshape_3d(ctx0, k, d_head, n_head, T2);
        q = ggml_rope_ext(ctx0, q, pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, d_head, GGML_ROPE_TYPE_NEOX, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3)); // [64, T2, 20]
        k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));
        v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, d_head, n_head, T2), 0, 2, 1, 3));
        ggml_tensor * scores = ggml_scale(ctx0, ggml_mul_mat(ctx0, k, q), 1.0f / sqrtf((float) d_head));
        ggml_tensor * probs = ggml_soft_max(ctx0, scores);
        ggml_tensor * o = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, v)), probs); // [64, T2, 20]
        o = ggml_cont(ctx0, ggml_permute(ctx0, o, 0, 2, 1, 3));
        o = ggml_reshape_2d(ctx0, o, n_head * d_head, T2);
        o = cbx_linear(ctx0, cbx_t(model, p + ".out.weight"), cbx_t(model, p + ".out.bias"), o);
        x = ggml_add(ctx0, res, ggml_add(ctx0, o, fsm));

        const std::string mp = "a.s3tok.encoder.blocks." + std::to_string(li) + ".mlp";
        res = x;
        cur = cbx_layer_norm(ctx0, cbx_t(model, mp + "_ln.weight"), cbx_t(model, mp + "_ln.bias"), x, 1e-5f);
        cur = cbx_linear(ctx0, cbx_t(model, mp + ".0.weight"), cbx_t(model, mp + ".0.bias"), cur);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = cbx_linear(ctx0, cbx_t(model, mp + ".2.weight"), cbx_t(model, mp + ".2.bias"), cur);
        x = ggml_add(ctx0, res, cur);
    }

    x = cbx_linear(ctx0, cbx_t(model, "a.s3tok.quantizer._codebook.project_down.weight"),
                   cbx_t(model, "a.s3tok.quantizer._codebook.project_down.bias"), x); // [8, T2]
    x = ggml_tanh(ctx0, x);

    ggml_set_name(x, "out_fsq");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

static ggml_cgraph * cbx_build_vocoder(const clip_model & model, ggml_context * ctx0, ggml_cgraph * gf, int n_mel, int n_stft);

ggml_cgraph * clip_graph_chatterbox::build() {
    if (gen_process == CLIP_GEN_PROCESS_TTS_VOCODE) {
        return cbx_build_vocoder(model, ctx0, gf, vocode_n_mel, vocode_n_stft);
    }
    if (gen_process == CLIP_GEN_PROCESS_TOKENIZE) {
        return cbx_build_s3tok(model, ctx0, gf, img.nx());
    }
    if (gen_process != CLIP_GEN_PROCESS_TTS) {
        // load-time buffer sizing path
        ggml_tensor * inp = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_set_name(inp, "inp_stub");
        ggml_set_input(inp);
        ggml_tensor * cur = ggml_dup(ctx0, inp);
        ggml_set_name(cur, "out_stub");
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
        return gf;
    }

    const int T1 = n_tokens;      // token-rate length (prompt + generated)
    const int T2 = 2 * n_tokens;  // mel-rate length after the x2 upsample

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T1);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    ggml_tensor * pos1 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, 2 * T1 - 1);
    ggml_set_name(pos1, "inp_pos1");
    ggml_set_input(pos1);

    ggml_tensor * pos2 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, 2 * T2 - 1);
    ggml_set_name(pos2, "inp_pos2");
    ggml_set_input(pos2);

    // token embedding
    ggml_tensor * x = ggml_get_rows(ctx0, cbx_t(model, "a.gen.flow.input_embedding.weight"), inp_tokens); // [512, T1]

    // embed: linear + layer norm, then the espnet xscale
    x = cbx_linear(ctx0, cbx_t(model, "a.gen.fenc.embed.out.0.weight"), cbx_t(model, "a.gen.fenc.embed.out.0.bias"), x);
    x = cbx_layer_norm(ctx0, cbx_t(model, "a.gen.fenc.embed.out.1.weight"), cbx_t(model, "a.gen.fenc.embed.out.1.bias"), x, 1e-5f);
    x = ggml_scale(ctx0, x, sqrtf(512.0f));
    cb(x, "fenc_embd", -1);

    // pre-lookahead: conv k=4 right-padded 3, leaky 0.01, conv k=3 left-padded 2, residual
    {
        ggml_tensor * res = x;
        ggml_tensor * cur = cbx_conv1d(ctx0, cbx_t(model, "a.gen.fenc.pre_lookahead_layer.conv1.weight"),
                                       cbx_t(model, "a.gen.fenc.pre_lookahead_layer.conv1.bias"), x, 1, 0, 3);
        cur = ggml_leaky_relu(ctx0, cur, 0.01f, false);
        cur = cbx_conv1d(ctx0, cbx_t(model, "a.gen.fenc.pre_lookahead_layer.conv2.weight"),
                         cbx_t(model, "a.gen.fenc.pre_lookahead_layer.conv2.bias"), cur, 1, 2, 0);
        x = ggml_add(ctx0, res, cur);
        cb(x, "fenc_pre_lookahead", -1);
    }

    for (int i = 0; model.cbx_tensors.count("a.gen.fenc.encoders." + std::to_string(i) + ".norm_mha.weight"); i++) {
        x = cbx_enc_layer(model, ctx0, x, pos1, "a.gen.fenc.encoders." + std::to_string(i), T1);
        cb(x, "fenc_enc", i);
    }

    // upsample x2: nearest repeat, left pad 4, conv k=5
    {
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));            // [T1, 512]
        xt = ggml_interpolate(ctx0, xt, 2 * T1, 512, 1, 1, GGML_SCALE_MODE_NEAREST);
        ggml_tensor * z = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 4, 512);
        z = ggml_scale(ctx0, z, 0.0f);
        xt = ggml_concat(ctx0, z, xt, 0);
        ggml_tensor * y = ggml_conv_1d(ctx0, cbx_t(model, "a.gen.fenc.up_layer.conv.weight"), xt, 1, 0, 1);
        x = ggml_cont(ctx0, ggml_transpose(ctx0, y));                            // [512, T2]
        x = ggml_add(ctx0, x, cbx_t(model, "a.gen.fenc.up_layer.conv.bias"));
        cb(x, "fenc_upsample", -1);
    }

    // up embed: linear + layer norm + xscale
    x = cbx_linear(ctx0, cbx_t(model, "a.gen.fenc.up_embed.out.0.weight"), cbx_t(model, "a.gen.fenc.up_embed.out.0.bias"), x);
    x = cbx_layer_norm(ctx0, cbx_t(model, "a.gen.fenc.up_embed.out.1.weight"), cbx_t(model, "a.gen.fenc.up_embed.out.1.bias"), x, 1e-5f);
    x = ggml_scale(ctx0, x, sqrtf(512.0f));

    for (int i = 0; model.cbx_tensors.count("a.gen.fenc.up_encoders." + std::to_string(i) + ".norm_mha.weight"); i++) {
        x = cbx_enc_layer(model, ctx0, x, pos2, "a.gen.fenc.up_encoders." + std::to_string(i), T2);
        cb(x, "fenc_up_enc", i);
    }

    x = cbx_layer_norm(ctx0, cbx_t(model, "a.gen.fenc.after_norm.weight"), cbx_t(model, "a.gen.fenc.after_norm.bias"), x, 1e-5f);

    // encoder projection to the mel channel count
    ggml_tensor * mu = cbx_linear(ctx0, cbx_t(model, "a.gen.flow.encoder_proj.weight"), cbx_t(model, "a.gen.flow.encoder_proj.bias"), x); // [80, T2]
    cb(mu, "flow_mu", -1);

    // cfm solver, unrolled in the graph. meanflow (distilled): 2 euler steps
    // over t = 0 -> 0.5 -> 1, no cfg, time embeds mix t and r. classic: 10
    // euler steps on the cosine schedule with cfg 0.7, time embeds on t only.
    const bool meanflow = model.cbx_tensors.count("a.gen.est.time_embed_mixer.weight") > 0;
    const int n_steps = meanflow ? 2 : 10;

    // span points, same schedule as the host side sinusoid fill in clip.cpp
    float span[11];
    for (int i = 0; i <= n_steps; i++) {
        const float u = (float) i / n_steps;
        span[i] = meanflow ? u : 1.0f - cosf(u * (float) M_PI / 2.0f);
    }

    ggml_tensor * noise = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, T2);
    ggml_set_name(noise, "inp_noise");
    ggml_set_input(noise);
    // sinusoidal time embeddings, one row per span point
    ggml_tensor * temb_sin = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 320, n_steps + 1);
    ggml_set_name(temb_sin, "inp_temb");
    ggml_set_input(temb_sin);

    auto time_mlp = [&](ggml_tensor * e) {
        e = cbx_linear(ctx0, cbx_t(model, "a.gen.est.time_mlp.linear_1.weight"), cbx_t(model, "a.gen.est.time_mlp.linear_1.bias"), e);
        e = ggml_silu(ctx0, e);
        e = cbx_linear(ctx0, cbx_t(model, "a.gen.est.time_mlp.linear_2.weight"), cbx_t(model, "a.gen.est.time_mlp.linear_2.bias"), e);
        return e;
    };
    auto span_emb = [&](int i) {
        return ggml_view_2d(ctx0, temb_sin, 320, 1, temb_sin->nb[1], (size_t) i * temb_sin->nb[1]);
    };
    auto step_temb = [&](int i) {
        if (!meanflow) {
            return time_mlp(span_emb(i));
        }
        ggml_tensor * e = ggml_concat(ctx0, time_mlp(span_emb(i)), time_mlp(span_emb(i + 1)), 0); // [2048, 1]
        return ggml_mul_mat(ctx0, cbx_t(model, "a.gen.est.time_embed_mixer.weight"), e);                // [1024, 1]
    };

    // mel-rate conditions: prompt features then zeros, and the 80-dim
    // speaker vector; both are fed by the host from either the reference
    // clip or the precomputed defaults
    ggml_tensor * pf = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, n_prompt_mel);
    ggml_set_name(pf, "inp_prompt_feat");
    ggml_set_input(pf);
    ggml_tensor * zc = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, T2 - n_prompt_mel);
    zc = ggml_scale(ctx0, zc, 0.0f);
    ggml_tensor * cond = ggml_concat(ctx0, pf, zc, 1); // [80, T2]
    ggml_tensor * spks = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 80);
    ggml_set_name(spks, "inp_spk");
    ggml_set_input(spks);

    const float cfg = 0.7f;
    ggml_tensor * mu_zero   = meanflow ? nullptr : ggml_scale(ctx0, mu, 0.0f);
    ggml_tensor * cond_zero = meanflow ? nullptr : ggml_scale(ctx0, cond, 0.0f);
    ggml_tensor * spks_zero = meanflow ? nullptr : ggml_scale(ctx0, spks, 0.0f);

    ggml_tensor * mx = noise;
    for (int i = 0; i < n_steps; i++) {
        ggml_tensor * temb = step_temb(i);
        ggml_tensor * d = cbx_estimator(model, ctx0, mx, mu, spks, cond, temb, T2);
        if (!meanflow) {
            ggml_tensor * du = cbx_estimator(model, ctx0, mx, mu_zero, spks_zero, cond_zero, temb, T2);
            d = ggml_add(ctx0, ggml_scale(ctx0, d, 1.0f + cfg), ggml_scale(ctx0, du, -cfg));
        }
        mx = ggml_add(ctx0, mx, ggml_scale(ctx0, d, span[i + 1] - span[i]));
        cb(mx, "cfm_step", i);
    }
    ggml_tensor * mel = mx;

    // trim the prompt frames at mel rate
    mel = ggml_view_2d(ctx0, mel, 80, T2 - n_prompt_mel, mel->nb[1], (size_t) n_prompt_mel * mel->nb[1]);
    mel = ggml_cont(ctx0, mel);
    ggml_set_name(mel, "out_mel");
    ggml_set_output(mel);
    ggml_build_forward_expand(gf, mel);

    // f0 predictor on the trimmed mel: 5x (conv k3 same-pad + elu), abs(linear)
    {
        ggml_tensor * fx = mel;
        for (int i = 0; model.cbx_tensors.count("a.gen.hift.f0_predictor.condnet." + std::to_string(i) + ".weight"); i += 2) {
            const std::string cp = "a.gen.hift.f0_predictor.condnet." + std::to_string(i);
            fx = cbx_conv1d(ctx0, cbx_t(model, cp + ".weight"), cbx_t(model, cp + ".bias"), fx, 1, 1, 1);
            fx = ggml_elu(ctx0, fx);
        }
        fx = cbx_linear(ctx0, cbx_t(model, "a.gen.hift.f0_predictor.classifier.weight"), cbx_t(model, "a.gen.hift.f0_predictor.classifier.bias"), fx);
        fx = ggml_abs(ctx0, fx); // [1, T]
        ggml_set_name(fx, "out_f0");
        ggml_set_output(fx);
        ggml_build_forward_expand(gf, fx);
    }
    return gf;
}


static ggml_tensor * cbx_hift_resblock(const clip_model & model, ggml_context * ctx0, ggml_tensor * x, const std::string & p);

// x [C, T] -> symmetric-padded dilated conv -> [OC, T]
ggml_tensor * cbx_conv1d_dil(ggml_context * ctx0, ggml_tensor * k, ggml_tensor * b, ggml_tensor * x, int pad, int dil) {
    ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * y = ggml_conv_1d(ctx0, k, xt, 1, pad, dil);
    y = ggml_cont(ctx0, ggml_transpose(ctx0, y));
    if (b) {
        y = ggml_add(ctx0, y, b);
    }
    return y;
}

// mel [80, T] + source stft [18, T_stft] -> conv_post output [18, T_stft2]
static ggml_cgraph * cbx_build_vocoder(const clip_model & model, ggml_context * ctx0, ggml_cgraph * gf, int n_mel, int n_stft) {
    ggml_tensor * mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 80, n_mel);
    ggml_set_name(mel, "inp_mel");
    ggml_set_input(mel);
    ggml_tensor * sstft = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 18, n_stft);
    ggml_set_name(sstft, "inp_sstft");
    ggml_set_input(sstft);

    ggml_tensor * x = cbx_conv1d_dil(ctx0, cbx_t(model, "a.gen.hift.conv_pre.weight"), cbx_t(model, "a.gen.hift.conv_pre.bias"), mel, 3, 1);

    for (int i = 0; model.cbx_tensors.count("a.gen.hift.ups." + std::to_string(i) + ".weight"); i++) {
        const std::string is = std::to_string(i);
        ggml_tensor * uk = cbx_t(model, "a.gen.hift.ups." + is + ".weight");
        const int K = (int) uk->ne[0];
        const int S = K / 2;
        const int P = (K - S) / 2;

        x = ggml_leaky_relu(ctx0, x, 0.1f, false);
        // conv transpose then trim the torch padding P on both sides
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));
        xt = ggml_conv_transpose_1d(ctx0, uk, xt, S, 0, 1);
        xt = ggml_cont(ctx0, ggml_view_2d(ctx0, xt, xt->ne[0] - 2 * P, xt->ne[1], xt->nb[1], (size_t) P * ggml_element_size(xt)));
        x = ggml_cont(ctx0, ggml_transpose(ctx0, xt));
        x = ggml_add(ctx0, x, cbx_t(model, "a.gen.hift.ups." + is + ".bias"));

        const bool is_last = !model.cbx_tensors.count("a.gen.hift.ups." + std::to_string(i + 1) + ".weight");
        if (is_last) {
            ggml_tensor * xr = ggml_cont(ctx0, ggml_transpose(ctx0, x));
            xr = ggml_pad_reflect_1d(ctx0, xr, 1, 0);
            x = ggml_cont(ctx0, ggml_transpose(ctx0, xr));
        }

        // source injection: strided conv on the source stft, one resblock
        ggml_tensor * sk = cbx_t(model, "a.gen.hift.source_downs." + is + ".weight");
        const int SK = (int) sk->ne[0];
        const int SS = SK > 1 ? SK / 2 : 1;
        const int SP = SK > 1 ? SS / 2 : 0;
        ggml_tensor * si;
        {
            ggml_tensor * st = ggml_cont(ctx0, ggml_transpose(ctx0, sstft));
            st = ggml_conv_1d(ctx0, sk, st, SS, SP, 1);
            si = ggml_cont(ctx0, ggml_transpose(ctx0, st));
            si = ggml_add(ctx0, si, cbx_t(model, "a.gen.hift.source_downs." + is + ".bias"));
        }
        si = cbx_hift_resblock(model, ctx0, si, "a.gen.hift.source_resblocks." + is);
        // align lengths: the reflection pad on the last stage adds one step
        if ((int) si->ne[1] != (int) x->ne[1]) {
            const int n = (int) std::min(si->ne[1], x->ne[1]);
            si = ggml_cont(ctx0, ggml_view_2d(ctx0, si, si->ne[0], n, si->nb[1], 0));
            x  = ggml_cont(ctx0, ggml_view_2d(ctx0, x,  x->ne[0],  n, x->nb[1],  0));
        }
        x = ggml_add(ctx0, x, si);

        ggml_tensor * acc = nullptr;
        for (int j = 3 * i; j < 3 * (i + 1); j++) {
            ggml_tensor * r = cbx_hift_resblock(model, ctx0, x, "a.gen.hift.resblocks." + std::to_string(j));
            acc = acc ? ggml_add(ctx0, acc, r) : r;
        }
        x = ggml_scale(ctx0, acc, 1.0f / 3.0f);
    }

    x = ggml_leaky_relu(ctx0, x, 0.01f, false);
    x = cbx_conv1d_dil(ctx0, cbx_t(model, "a.gen.hift.conv_post.weight"), cbx_t(model, "a.gen.hift.conv_post.bias"), x, 3, 1);
    ggml_set_name(x, "out_spec");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// snake activation with per-channel alpha: x + sin^2(alpha x) / (alpha + eps)
static ggml_tensor * cbx_snake(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * alpha) {
    ggml_tensor * sx = ggml_sin(ctx0, ggml_mul(ctx0, x, alpha));
    sx = ggml_mul(ctx0, sx, sx);
    sx = ggml_div(ctx0, sx, alpha);
    return ggml_add(ctx0, x, sx);
}

// hifigan-snake resblock; kernels with dilations 1/3/5 on convs1, 1 on convs2
static ggml_tensor * cbx_hift_resblock(const clip_model & model, ggml_context * ctx0, ggml_tensor * x, const std::string & p) {
    static const int dil[3] = {1, 3, 5};
    for (int j = 0; model.cbx_tensors.count(p + ".convs1." + std::to_string(j) + ".weight"); j++) {
        const std::string js = std::to_string(j);
        ggml_tensor * a1 = cbx_t(model, p + ".activations1." + js + ".alpha");
        ggml_tensor * a2 = cbx_t(model, p + ".activations2." + js + ".alpha");
        ggml_tensor * k1 = cbx_t(model, p + ".convs1." + js + ".weight");
        ggml_tensor * k2 = cbx_t(model, p + ".convs2." + js + ".weight");
        const int d = dil[j % 3];
        const int p1 = (int) (k1->ne[0] - 1) / 2 * d;
        const int p2 = (int) (k2->ne[0] - 1) / 2;
        ggml_tensor * xt = cbx_snake(ctx0, x, a1);
        xt = cbx_conv1d_dil(ctx0, k1, cbx_t(model, p + ".convs1." + js + ".bias"), xt, p1, d);
        xt = cbx_snake(ctx0, xt, a2);
        xt = cbx_conv1d_dil(ctx0, k2, cbx_t(model, p + ".convs2." + js + ".bias"), xt, p2, 1);
        x = ggml_add(ctx0, x, xt);
    }
    return x;
}

