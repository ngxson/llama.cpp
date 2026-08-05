#include "models.h"

// SEANet convolution stack of the mimi codec, see pocket_tts/modules/seanet.py
//
// tensors are T-first here: [T, C]. the convs are causal: they take left context from a
// state slot when given, otherwise they pad (cold start / one-shot encode)

static int64_t div_ceil(int64_t a, int64_t b) {
    return a / b + (a % b ? 1 : 0);
}

// x: [T, IC], w: [K, IC, OC] -> [T / stride, OC]
// the convs are causal, so the whole K - stride padding goes on the left
ggml_tensor * clip_graph_pockettts_seanet::conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, int dilation,
                                                  bool pad_replicate, const std::string & state_name) const {
    const int64_t k_size  = (w->ne[0] - 1) * dilation + 1;
    const int64_t p_total = k_size - stride;

    // trailing padding so the last frame is not dropped, see pad_for_conv1d() in conv.py
    const int64_t n_frames  = div_ceil(x->ne[0] - k_size + p_total, stride);
    const int64_t ideal_len = n_frames * stride + k_size - p_total;
    const int64_t p_extra   = ideal_len - x->ne[0];

    if (!state_name.empty() && p_total > 0) {
        // streaming: the left context is the tail of the previous call
        ggml_tensor * left = state_in.at(state_name); // [p_total, IC]
        x = ggml_concat(ctx0, left, x, 0);
        state_out.push_back({state_name,
            ggml_cont(ctx0, ggml_view_2d(ctx0, x, p_total, x->ne[1], x->nb[1],
                                         (size_t) (x->ne[0] - p_total) * x->nb[0]))});
    } else if (pad_replicate && p_total > 0) {
        // the resamplers repeat the first frame instead of zero-padding
        ggml_tensor * first = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], 0);
        ggml_tensor * left  = ggml_repeat_4d(ctx0, first, p_total, x->ne[1], 1, 1);
        x = ggml_concat(ctx0, left, x, 0);
        x = ggml_pad_ext(ctx0, x, 0, p_extra, 0, 0, 0, 0, 0, 0);
    } else {
        x = ggml_pad_ext(ctx0, x, p_total, p_extra, 0, 0, 0, 0, 0, 0);
    }

    ggml_tensor * y = ggml_conv_1d(ctx0, w, x, stride, 0, dilation);
    y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return y;
}

// x: [T, IC], w: [K, OC/groups, IC] -> [T * stride, OC]
// the K - stride overlap tail belongs to the next call: it is added to the head of the next
// output when streaming, and simply dropped otherwise
ggml_tensor * clip_graph_pockettts_seanet::conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride,
                                                            const std::string & state_name) const {
    const int64_t p_total   = w->ne[0] - stride;
    const bool    depthwise = w->ne[1] == 1 && w->ne[2] > 1;
    const int64_t emit_len  = x->ne[0] * stride;

    ggml_tensor * full = nullptr;
    if (depthwise) {
        // one group per channel, ggml_conv_transpose_1d has no grouped mode
        for (int64_t ir = 0; ir < x->ne[1]; ir++) {
            ggml_tensor * row = ggml_view_1d(ctx0, x, x->ne[0], ir * x->ne[0] * ggml_element_size(x));
            ggml_tensor * krn = ggml_view_1d(ctx0, w, w->ne[0], ir * w->ne[0] * ggml_element_size(w));
            row  = ggml_conv_transpose_1d(ctx0, krn, row, stride, 0, 1);
            full = full ? ggml_concat(ctx0, full, row, 1) : row;
        }
    } else {
        full = ggml_conv_transpose_1d(ctx0, w, x, stride, 0, 1);
    }
    full = ggml_cont(ctx0, full); // [emit_len + p_total, OC]

    ggml_tensor * out;
    if (state_name.empty() || p_total == 0) {
        out = ggml_cont(ctx0, ggml_view_2d(ctx0, full, emit_len, full->ne[1], full->nb[1], 0));
    } else {
        // overlap-add the tail the previous call held back
        ggml_tensor * prev = state_in.at(state_name); // [p_total, OC]
        ggml_tensor * head = ggml_add(ctx0, ggml_view_2d(ctx0, full, p_total, full->ne[1], full->nb[1], 0), prev);
        if (emit_len > p_total) {
            ggml_tensor * rest = ggml_view_2d(ctx0, full, emit_len - p_total, full->ne[1], full->nb[1],
                                              (size_t) p_total * full->nb[0]);
            out = ggml_concat(ctx0, head, rest, 0);
        } else {
            out = head;
        }
        state_out.push_back({state_name,
            ggml_cont(ctx0, ggml_view_2d(ctx0, full, p_total, full->ne[1], full->nb[1],
                                         (size_t) emit_len * full->nb[0]))});
    }

    if (b) {
        out = ggml_add(ctx0, out, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return out;
}

// ELU -> dilated conv -> ELU -> pointwise conv, added back to the input
ggml_tensor * clip_graph_pockettts_seanet::res_unit(ggml_tensor * x, const clip_seanet::stage & stage, int dilation,
                                                    const std::string & state_prefix) const {
    ggml_tensor * h = ggml_elu(ctx0, x);
    h = conv1d(h, stage.res_conv1_w, stage.res_conv1_b, 1, dilation, false, state_prefix);
    h = ggml_elu(ctx0, h);
    // the second conv is pointwise, it needs no left context
    h = conv1d(h, stage.res_conv2_w, stage.res_conv2_b, 1, 1);
    return ggml_add(ctx0, x, h);
}

ggml_tensor * clip_graph_pockettts_seanet::encode(ggml_tensor * x) const {
    const auto & seanet = model.seanet;

    ggml_tensor * cur = conv1d(x, seanet.conv_in_w, seanet.conv_in_b, 1, 1);
    cb(cur, "seanet_enc_in", -1);

    for (int i = 0; i < hparams.seanet_n_stage; i++) {
        const auto & stage  = seanet.stages[i];
        const int    stride = hparams.seanet_ratios[i];

        cur = res_unit(cur, stage, 1);
        cur = ggml_elu(ctx0, cur);
        cur = conv1d(cur, stage.scale_conv_w, stage.scale_conv_b, stride, 1);
        cb(cur, "seanet_enc_stage", i);
    }

    cur = ggml_elu(ctx0, cur);
    cur = conv1d(cur, seanet.conv_out_w, seanet.conv_out_b, 1, 1);
    cb(cur, "seanet_enc_out", -1);

    return cur;
}

ggml_tensor * clip_graph_pockettts_seanet::decode(ggml_tensor * x) const {
    const auto & seanet = model.seanet;
    const bool   stream = !state_in.empty();

    ggml_tensor * cur = conv1d(x, seanet.conv_in_w, seanet.conv_in_b, 1, 1, false,
                               stream ? "dec_in" : "");
    cb(cur, "seanet_dec_in", -1);

    for (int i = 0; i < hparams.seanet_n_stage; i++) {
        const auto & stage = seanet.stages[i];
        // the decoder mirrors the encoder, so the ratios are walked backwards
        const int    stride = hparams.seanet_ratios[hparams.seanet_n_stage - 1 - i];
        const std::string id = std::to_string(i);

        cur = ggml_elu(ctx0, cur);
        cur = conv_transpose1d(cur, stage.scale_conv_w, stage.scale_conv_b, stride,
                               stream ? "dec_up_" + id : "");
        cur = res_unit(cur, stage, 1, stream ? "dec_res_" + id : "");
        cb(cur, "seanet_dec_stage", i);
    }

    cur = ggml_elu(ctx0, cur);
    cur = conv1d(cur, seanet.conv_out_w, seanet.conv_out_b, 1, 1, false,
                 stream ? "dec_out" : "");
    cb(cur, "seanet_dec_out", -1);

    return cur;
}
