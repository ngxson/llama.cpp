#pragma once

#include "clip.h"
#include "clip-impl.h"
#include "ggml.h"

#include <cmath>
#include <vector>
#include <map>

using get_tensor_fn = std::function<ggml_tensor * (const std::string & name)>;

static ggml_tensor * conv2d(ggml_context * ctx, ggml_tensor * kernel, ggml_tensor * inp, int strides, bool depthwise = false) {
    int p0 = 0;
    int p1 = 0;

    {
        const int kernel_size = kernel->ne[0];

        auto compute_padding_length = [](int input_length, int kernel_length, int stride) {
            int total_padding_length = (kernel_length - 1) - (input_length - 1) % stride;
            int left_padding = total_padding_length / 2;
            int right_padding = (total_padding_length + 1) / 2;
            return std::make_pair(left_padding, right_padding);
        };

        auto [left, right] = compute_padding_length(inp->ne[0], kernel_size, strides);
        auto [top, bottom] = compute_padding_length(inp->ne[1], kernel_size, strides);

        if (left > 0 && right > 0) {
            p0 = std::min(left, right);
            left -= p0;
            right -= p0;
        }

        if (top > 0 && bottom > 0) {
            p1 = std::min(top, bottom);
            top -= p1;
            bottom -= p1;
        }

        GGML_ASSERT(left == 0 && top == 0);

        if (right != 0 || bottom != 0) {
            inp = ggml_pad(ctx, inp, right, bottom, 0, 0);
        }
    }

    ggml_tensor * cur;

    if (depthwise) {
        cur = ggml_conv_2d_dw(ctx,
            kernel, inp,
            strides, strides,
            p0, p1, 1, 1);
    } else {
        cur = ggml_conv_2d(ctx,
            kernel, inp,
            strides, strides,
            p0, p1, 1, 1);
    }

    return cur;
}

struct mobilenet_g3n_blk {
    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur) = 0;
    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) = 0;
    virtual ~mobilenet_g3n_blk() = default;
};

// ConvNormAct
struct mobilenet_g3n_cna : mobilenet_g3n_blk {
    int kernel_size = 0;
    int stride = 1;
    int dilation = 1;
    int filters = 0;
    float expand_ratio = 1.0f;

    ggml_tensor * norm = nullptr;
    ggml_tensor * conv = nullptr;

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
        std::string tmp;
        tmp = prefix + "bn.weight";
        norm = get_tensor(tmp);
        tmp = prefix + "conv.weight";
        conv = get_tensor(tmp);
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur) {
        cur = conv2d(ctx, conv, cur, stride, true);
        cur = ggml_group_norm(ctx, cur, std::min<int>(32, filters * expand_ratio / 4), 1e-6f);
        cur = ggml_mul(ctx, cur, ggml_reshape_3d(ctx, norm, 1, 1, norm->ne[0]));
        return cur;
    }
};

// EdgeResidual
struct mobilenet_g3n_er : mobilenet_g3n_blk {
    int kernel_size = 0;
    int stride = 1;
    int filters = 0;

    ggml_tensor * norm1 = nullptr;
    ggml_tensor * norm2 = nullptr;
    ggml_tensor * conv_exp = nullptr;
    ggml_tensor * conv_pwl = nullptr;

    mobilenet_g3n_er(int kernel_size, int filters, int stride) :
        kernel_size(kernel_size), stride(stride), filters(filters) {}

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
        std::string tmp;
        tmp = prefix + "bn1.weight";
        norm1 = get_tensor(tmp);
        tmp = prefix + "bn2.weight";
        norm2 = get_tensor(tmp);
        tmp = prefix + "conv_exp.weight";
        conv_exp = get_tensor(tmp);
        tmp = prefix + "conv_pwl.weight";
        conv_pwl = get_tensor(tmp);
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur) {
        return cur;
    }
};

// UniversalInvertedResidual
struct mobilenet_g3n_uir : mobilenet_g3n_blk {
    int start_dw_kernel_size = 0;
    int mid_dw_kernel_size = 0;
    bool multiscale = false;
    ggml_tensor * layer_scale = nullptr;

    mobilenet_g3n_cna dw_start;
    mobilenet_g3n_cna dw_mid;
    mobilenet_g3n_cna dw_end;
    mobilenet_g3n_cna dw_proj;

    mobilenet_g3n_uir(int start_dw_kernel_size, int mid_dw_kernel_size, int filters, int stride = 1, float expand_ratio = 4.0f, bool multiscale = false) :
            start_dw_kernel_size(start_dw_kernel_size),
            mid_dw_kernel_size(mid_dw_kernel_size),
            multiscale(multiscale) {
        dw_start.stride = stride;
        dw_start.filters = filters;
        dw_start.expand_ratio = expand_ratio;

        dw_mid.stride = 1;
        dw_mid.filters = filters;
        dw_mid.expand_ratio = expand_ratio;

        dw_end.stride = 1;
        dw_end.filters = filters;
        dw_end.expand_ratio = expand_ratio;

        dw_proj.stride = 1;
        dw_proj.filters = filters;
        dw_proj.expand_ratio = 1.0f; // projection does not expand
    }

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
        dw_start.load_tensors(prefix + "dw_start.", get_tensor);
        dw_mid.load_tensors(prefix + "dw_mid.", get_tensor);
        dw_end.load_tensors(prefix + "dw_end.", get_tensor);
        dw_proj.load_tensors(prefix + "dw_proj.", get_tensor);
        layer_scale = get_tensor(prefix + "layer_scale.weight");
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur) {
        if (dw_start.conv) {
            cur = dw_start.build(ctx, cur);
        }
        if (dw_mid.conv) {
            cur = dw_mid.build(ctx, cur);
        }
        if (dw_end.conv) {
            cur = dw_end.build(ctx, cur);
        }
        if (dw_proj.conv) {
            cur = dw_proj.build(ctx, cur);
        }

        cur = ggml_mul(ctx, cur, ggml_reshape_3d(ctx, layer_scale, 1, 1, layer_scale->ne[0]));

        return cur;
    }
};

// MultiQueryAttentionBlock
struct mobilenet_g3n_mmqa : mobilenet_g3n_blk {
    int num_heads = 0;
    int kv_strides = 0;
    int kv_dim = 0;
    bool mmqa_avg_pool_kv = false;
    bool multiscale = false;

    mobilenet_g3n_mmqa(int num_heads, int kv_dim, int kv_strides, 
                       bool mmqa_avg_pool_kv = false, bool multiscale = false) :
        num_heads(num_heads), kv_dim(kv_dim), kv_strides(kv_strides),
        mmqa_avg_pool_kv(mmqa_avg_pool_kv), multiscale(multiscale) {}

    virtual void load_tensors(const std::string & prefix, get_tensor_fn & get_tensor) {
    }

    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur) {
        return cur;
    }
};

struct mobilenet_g3n {
    // mapping prefix to block, order is important
    std::map<mobilenet_g3n_blk *, std::string> blocks;

    // temporary variables
    int stg_idx = 0;
    int blk_idx = 0;

    mobilenet_g3n() {
        // Stage 1: Edge Residuals
        stg_idx = 0; blk_idx = 0;
        add(    new mobilenet_g3n_er(3, 128, 2));
        add(2,  new mobilenet_g3n_er(3, 128, 1));

        // Stage 2: Universal Inverted Residuals
        stg_idx = 1; blk_idx = 0;
        add(    new mobilenet_g3n_uir(3, 5, 256, 2, 6.0f));
        add(    new mobilenet_g3n_uir(5, 0, 256));
        add(    new mobilenet_g3n_uir(3, 0, 256));
        add(    new mobilenet_g3n_uir(5, 0, 256));
        add(    new mobilenet_g3n_uir(3, 0, 256));

        // Stage 3: Universal Inverted Residuals with Multi-Query Attention
        stg_idx = 2; blk_idx = 0;
        add(    new mobilenet_g3n_uir(5, 5, 640, 2, 6.0f));
        add(7,  new mobilenet_g3n_uir(5, 0, 640));
        add(    new mobilenet_g3n_uir(0, 0, 640, 1, 1.0f));
        add(13, new mobilenet_g3n_mmqa(12, 64, 2), new mobilenet_g3n_uir(0, 0, 640, 1, 2.0f));
        add(    new mobilenet_g3n_mmqa(12, 64, 2), new mobilenet_g3n_uir(0, 0, 640, 1, 2.0f, true));

        // Stage 4: Universal Inverted Residuals with Multi-Query Attention
        stg_idx = 3; blk_idx = 0;
        add(    new mobilenet_g3n_uir(5, 5, 1280, 2, 6.0f));
        add(18, new mobilenet_g3n_mmqa(16, 96, 1), new mobilenet_g3n_uir(0, 0, 1280, 1, 2.0f));
        add(    new mobilenet_g3n_mmqa(16, 96, 1), new mobilenet_g3n_uir(0, 0, 1280, 1, 2.0f, true));
    }

    ~mobilenet_g3n() {
        for (auto & blk : blocks) {
            delete blk.first;
        }
    }

    void load_tensors(get_tensor_fn & get_tensor) {
        for (auto blk : blocks) {
            blk.first->load_tensors(blk.second, get_tensor);
        }
    }

    ggml_tensor * build(ggml_context * ctx, ggml_tensor * cur) {
        return cur;
    }

    void add(mobilenet_g3n_blk * blk) {
        blocks.insert({ blk, string_format("v.mobilenet.%d.%d.", stg_idx, blk_idx++) });
    }

    void add(mobilenet_g3n_blk * blk0, mobilenet_g3n_blk * blk1) {
        add(blk0);
        add(blk1);
    }

    void add(int count, mobilenet_g3n_blk * blk) {
        for (int i = 0; i < count; ++i) {
            add(blk);
        }
    }

    void add(int count, mobilenet_g3n_blk * blk0, mobilenet_g3n_blk * blk1) {
        for (int i = 0; i < count; ++i) {
            add(blk0, blk1);
        }
    }
};
