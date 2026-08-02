#include "models.h"

#include <string>

// Chatterbox speaker encoder: CAMPPlus x-vector on kaldi fbank features,
// projected through the s3gen speaker affine. Mirrors s3gen/xvector.py.

// per-channel batchnorm on x [C, T]; scale = w / sqrt(var + eps), shift folds
// the running mean. pass null w/b for the affine=False variant
static ggml_tensor * cbx_bn1d(const clip_model & model, ggml_context * ctx0, ggml_tensor * x,
                              const std::string & p, ggml_tensor * eps) {
    ggml_tensor * mean = cbx_t(model, p + ".running_mean");
    ggml_tensor * var  = cbx_t(model, p + ".running_var");
    ggml_tensor * sd   = ggml_sqrt(ctx0, ggml_add(ctx0, var, eps));
    if (!model.cbx_tensors.count(p + ".weight")) {
        return ggml_div(ctx0, ggml_sub(ctx0, x, mean), sd);
    }
    ggml_tensor * a     = ggml_div(ctx0, cbx_t(model, p + ".weight"), sd);
    ggml_tensor * shift = ggml_sub(ctx0, cbx_t(model, p + ".bias"), ggml_mul(ctx0, mean, a));
    return ggml_add(ctx0, ggml_mul(ctx0, x, a), shift);
}

// batchnorm + relu on a conv2d activation [W=T, H=F, C, 1], stats on ne2
static ggml_tensor * cbx_bn2d_relu(const clip_model & model, ggml_context * ctx0, ggml_tensor * x,
                                   const std::string & p, ggml_tensor * eps) {
    const int C = (int) x->ne[2];
    ggml_tensor * mean = ggml_reshape_4d(ctx0, cbx_t(model, p + ".running_mean"), 1, 1, C, 1);
    ggml_tensor * var  = ggml_reshape_4d(ctx0, cbx_t(model, p + ".running_var"),  1, 1, C, 1);
    ggml_tensor * w    = ggml_reshape_4d(ctx0, cbx_t(model, p + ".weight"), 1, 1, C, 1);
    ggml_tensor * b    = ggml_reshape_4d(ctx0, cbx_t(model, p + ".bias"),   1, 1, C, 1);
    ggml_tensor * a     = ggml_div(ctx0, w, ggml_sqrt(ctx0, ggml_add(ctx0, var, eps)));
    ggml_tensor * shift = ggml_sub(ctx0, b, ggml_mul(ctx0, mean, a));
    return ggml_relu(ctx0, ggml_add(ctx0, ggml_mul(ctx0, x, a), shift));
}

// fcm residual 2d block, stride on the frequency axis only
static ggml_tensor * cbx_res2d(const clip_model & model, ggml_context * ctx0, ggml_tensor * x,
                               const std::string & p, int stride, ggml_tensor * eps) {
    ggml_tensor * cur = ggml_conv_2d(ctx0, cbx_t(model, p + ".conv1.weight"), x, 1, stride, 1, 1, 1, 1);
    cur = cbx_bn2d_relu(model, ctx0, cur, p + ".bn1", eps);
    cur = ggml_conv_2d(ctx0, cbx_t(model, p + ".conv2.weight"), cur, 1, 1, 1, 1, 1, 1);
    // bn2 without the relu, applied before the residual add
    {
        const int C = (int) cur->ne[2];
        ggml_tensor * mean = ggml_reshape_4d(ctx0, cbx_t(model, p + ".bn2.running_mean"), 1, 1, C, 1);
        ggml_tensor * var  = ggml_reshape_4d(ctx0, cbx_t(model, p + ".bn2.running_var"),  1, 1, C, 1);
        ggml_tensor * w    = ggml_reshape_4d(ctx0, cbx_t(model, p + ".bn2.weight"), 1, 1, C, 1);
        ggml_tensor * b    = ggml_reshape_4d(ctx0, cbx_t(model, p + ".bn2.bias"),   1, 1, C, 1);
        ggml_tensor * a     = ggml_div(ctx0, w, ggml_sqrt(ctx0, ggml_add(ctx0, var, eps)));
        ggml_tensor * shift = ggml_sub(ctx0, b, ggml_mul(ctx0, mean, a));
        cur = ggml_add(ctx0, ggml_mul(ctx0, cur, a), shift);
    }
    ggml_tensor * res = x;
    if (model.cbx_tensors.count(p + ".shortcut.0.weight")) {
        res = ggml_conv_2d(ctx0, cbx_t(model, p + ".shortcut.0.weight"), x, 1, stride, 0, 0, 1, 1);
        const int C = (int) res->ne[2];
        ggml_tensor * mean = ggml_reshape_4d(ctx0, cbx_t(model, p + ".shortcut.1.running_mean"), 1, 1, C, 1);
        ggml_tensor * var  = ggml_reshape_4d(ctx0, cbx_t(model, p + ".shortcut.1.running_var"),  1, 1, C, 1);
        ggml_tensor * w    = ggml_reshape_4d(ctx0, cbx_t(model, p + ".shortcut.1.weight"), 1, 1, C, 1);
        ggml_tensor * b    = ggml_reshape_4d(ctx0, cbx_t(model, p + ".shortcut.1.bias"),   1, 1, C, 1);
        ggml_tensor * a     = ggml_div(ctx0, w, ggml_sqrt(ctx0, ggml_add(ctx0, var, eps)));
        ggml_tensor * shift = ggml_sub(ctx0, b, ggml_mul(ctx0, mean, a));
        res = ggml_add(ctx0, ggml_mul(ctx0, res, a), shift);
    }
    return ggml_relu(ctx0, ggml_add(ctx0, cur, res));
}

// cam dense tdnn layer: bottleneck then context-gated conv; x [C_in, T] -> [growth, T]
static ggml_tensor * cbx_cam_layer(const clip_model & model, ggml_context * ctx0, ggml_tensor * x,
                                   const std::string & p, int dil, ggml_tensor * eps, ggml_tensor * segfix) {
    ggml_tensor * h = ggml_relu(ctx0, cbx_bn1d(model, ctx0, x, p + ".nonlinear1.batchnorm", eps));
    h = cbx_conv1d(ctx0, cbx_t(model, p + ".linear1.weight"), nullptr, h, 1, 0, 0);
    h = ggml_relu(ctx0, cbx_bn1d(model, ctx0, h, p + ".nonlinear2.batchnorm", eps));

    const std::string cp = p + ".cam_layer";
    ggml_tensor * k = cbx_t(model, cp + ".linear_local.weight");
    const int pad = ((int) k->ne[0] - 1) / 2 * dil;
    ggml_tensor * y = cbx_conv1d_dil(ctx0, k, nullptr, h, pad, dil);

    // context: global mean plus ceil-mode segment means of length 100
    const int T = (int) h->ne[1];
    const int C = (int) h->ne[0];
    const int S = (T + 99) / 100;
    ggml_tensor * ht = ggml_cont(ctx0, ggml_transpose(ctx0, h)); // [T, C]
    ggml_tensor * gmean = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_mean(ctx0, ht))); // [C, 1]
    ggml_tensor * seg;
    {
        ggml_tensor * padded = ht;
        if (S * 100 != T) {
            ggml_tensor * z = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, S * 100 - T, C);
            z = ggml_scale(ctx0, z, 0.0f);
            padded = ggml_concat(ctx0, ht, z, 0);
        }
        ggml_tensor * pooled = ggml_pool_1d(ctx0, padded, GGML_OP_POOL_AVG, 100, 100, 0); // [S, C]
        pooled = ggml_mul(ctx0, pooled, segfix);
        ggml_tensor * exp = ggml_interpolate(ctx0, pooled, S * 100, C, 1, 1, GGML_SCALE_MODE_NEAREST);
        exp = ggml_cont(ctx0, ggml_view_2d(ctx0, exp, T, C, exp->nb[1], 0));
        seg = ggml_cont(ctx0, ggml_transpose(ctx0, exp));        // [C, T]
    }
    ggml_tensor * context = ggml_add(ctx0, seg, gmean);
    context = cbx_conv1d(ctx0, cbx_t(model, cp + ".linear1.weight"), cbx_t(model, cp + ".linear1.bias"), context, 1, 0, 0);
    context = ggml_relu(ctx0, context);
    context = cbx_conv1d(ctx0, cbx_t(model, cp + ".linear2.weight"), cbx_t(model, cp + ".linear2.bias"), context, 1, 0, 0);
    ggml_tensor * m = ggml_sigmoid(ctx0, context);
    return ggml_mul(ctx0, y, m);
}

ggml_cgraph * clip_graph_chatterbox_spkenc::build() {
    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(eps, "inp_eps");
    ggml_set_input(eps);

    const int T  = img.nx();
    const int T1 = (T - 1) / 2 + 1;
    const int S  = (T1 + 99) / 100;
    ggml_tensor * segfix = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, S);
    ggml_set_name(segfix, "inp_segfix");
    ggml_set_input(segfix);

    // fbank features [T, 80] from the preprocessor
    ggml_tensor * inp = build_inp_raw(1);

    // fcm 2d front: [W=T, H=F=80, C=1] -> [T, 10, 32] -> [320, T]
    ggml_tensor * x = ggml_reshape_4d(ctx0, inp, T, 80, 1, 1);
    x = ggml_conv_2d(ctx0, cbx_t(model, "spk.head.conv1.weight"), x, 1, 1, 1, 1, 1, 1);
    x = cbx_bn2d_relu(model, ctx0, x, "spk.head.bn1", eps);
    x = cbx_res2d(model, ctx0, x, "spk.head.layer1.0", 2, eps);
    x = cbx_res2d(model, ctx0, x, "spk.head.layer1.1", 1, eps);
    x = cbx_res2d(model, ctx0, x, "spk.head.layer2.0", 2, eps);
    x = cbx_res2d(model, ctx0, x, "spk.head.layer2.1", 1, eps);
    x = ggml_conv_2d(ctx0, cbx_t(model, "spk.head.conv2.weight"), x, 1, 2, 1, 1, 1, 1);
    x = cbx_bn2d_relu(model, ctx0, x, "spk.head.bn2", eps);
    x = ggml_reshape_2d(ctx0, x, T, 320);
    x = ggml_cont(ctx0, ggml_transpose(ctx0, x));               // [320, T]
    cb(x, "spk_fcm", -1);

    // tdnn k5 stride 2 over time, then the three cam dense blocks
    x = cbx_conv1d(ctx0, cbx_t(model, "spk.xvector.tdnn.linear.weight"), nullptr, x, 2, 2, 2); // [128, T1]
    x = ggml_relu(ctx0, cbx_bn1d(model, ctx0, x, "spk.xvector.tdnn.nonlinear.batchnorm", eps));
    cb(x, "spk_tdnn", -1);

    static const int block_dil[3] = {1, 2, 2};
    for (int bi = 1; bi <= 3; bi++) {
        const std::string bp = "spk.xvector.block" + std::to_string(bi);
        for (int li = 1; model.cbx_tensors.count(bp + ".tdnnd" + std::to_string(li) + ".linear1.weight"); li++) {
            ggml_tensor * out = cbx_cam_layer(model, ctx0, x, bp + ".tdnnd" + std::to_string(li),
                                              block_dil[bi - 1], eps, segfix);
            x = ggml_concat(ctx0, x, out, 0);
        }
        const std::string tp = "spk.xvector.transit" + std::to_string(bi);
        x = ggml_relu(ctx0, cbx_bn1d(model, ctx0, x, tp + ".nonlinear.batchnorm", eps));
        x = cbx_conv1d(ctx0, cbx_t(model, tp + ".linear.weight"), nullptr, x, 1, 0, 0);
        cb(x, "spk_block", bi);
    }
    x = ggml_relu(ctx0, cbx_bn1d(model, ctx0, x, "spk.xvector.out_nonlinear.batchnorm", eps)); // [512, T1]

    // statistics pooling: mean and unbiased std over time -> [1024, 1]
    ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));                 // [T1, 512]
    ggml_tensor * mean = ggml_mean(ctx0, xt);                                    // [1, 512]
    ggml_tensor * m2   = ggml_mean(ctx0, ggml_mul(ctx0, xt, xt));
    ggml_tensor * var  = ggml_sub(ctx0, m2, ggml_mul(ctx0, mean, mean));
    var = ggml_scale(ctx0, var, (float) T1 / (float) (T1 - 1));
    ggml_tensor * sd = ggml_sqrt(ctx0, ggml_relu(ctx0, var));
    ggml_tensor * stats = ggml_concat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, mean)),
                                            ggml_cont(ctx0, ggml_transpose(ctx0, sd)), 0); // [1024, 1]
    cb(stats, "spk_stats_pool", -1);

    // dense 1024 -> 192, batchnorm without affine, into the x-vector
    ggml_tensor * dw = ggml_reshape_2d(ctx0, cbx_t(model, "spk.xvector.dense.linear.weight"), 1024, 192);
    ggml_tensor * emb = ggml_mul_mat(ctx0, dw, stats);                           // [192, 1]
    emb = cbx_bn1d(model, ctx0, emb, "spk.xvector.dense.nonlinear.batchnorm", eps);
    emb = ggml_reshape_1d(ctx0, emb, 192);
    ggml_set_name(emb, "out_xvec");
    ggml_set_output(emb);
    ggml_build_forward_expand(gf, emb);

    // normalize then the s3gen speaker affine
    ggml_tensor * n2 = ggml_sqrt(ctx0, ggml_sum(ctx0, ggml_mul(ctx0, emb, emb)));
    ggml_tensor * unit = ggml_div(ctx0, emb, n2);
    ggml_tensor * spk80 = cbx_linear(ctx0, cbx_t(model, "flow.spk_embed_affine_layer.weight"),
                                     cbx_t(model, "flow.spk_embed_affine_layer.bias"),
                                     ggml_reshape_2d(ctx0, unit, 192, 1));
    spk80 = ggml_cont(ctx0, ggml_reshape_1d(ctx0, spk80, 80));
    cb(spk80, "spk_embd", -1);
    ggml_build_forward_expand(gf, spk80);
    return gf;
}
