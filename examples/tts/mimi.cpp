#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include "common.h"

#include <limits.h>
#include <vector>
#include <cinttypes>
#include <fstream>
#include <unordered_map>
#include <float.h>

/**
 * Implementation of Kyutai's Mimi model using GGML.
 * Based on this research: https://github.com/ngxson/ggml-easy/blob/master/demo/kyutai-mimi.cpp
 *
 * NOTE: only decoder is working for now.
 *
 * Background:
 * - The audio codes can be generated using any Mimi-based model, for example: Moshi, Hibiki, Sesame, etc
 * - Audio codes must be in the order: (1 semantic component, 31 acoustic components) repeated N times
 *
 * How it works?
 * 1. Audio code passed to RVQ (mimi_residual_vector_quantizer) to get the latent code
 * 2. The latent code is passed to a mimi_conv_transpose_1d (depthwise) to upscale
 * 3. The upscaled code is passed to transformer, it converts N frames to N frames
 * 4. The output embeddings is then passed to SEANet (mimi_encoder_decoder) to get the final waveform
 * 5. Waveform is written to a file
 */

// copied from https://huggingface.co/kyutai/mimi/blob/main/config.json
struct mimi_config_t {
    bool causal = true;
    int max_position_embeddings = 8000;
    int num_hidden_layers = 8;
    int n_embd = 512;
    int n_ffn = 2048;
    int n_head = 8;
    int n_head_kv = 8;
    int n_rot = 64;
    float norm_eps = 1e-5;
    float rope_theta = 10000.0f;
    int sliding_window = 250;
    std::array<int, 4> upsampling_ratio   = {8, 6, 5, 4};
    std::array<int, 4> downsampling_ratio = {4, 5, 6, 8}; // reverse of upsampling_ratio
    // vector quantizer
    float frame_rate = 12.5;
    int audio_channels = 1;
    int codebook_size = 2048;
    int codebook_dim = 256;
    int n_semantic_components = 1;
    int n_acoustic_components = 31;
    // decode
    float trim_right_ratio = 1.0f;
} mimi_config;

// Adapted from https://github.com/ngxson/ggml-easy/blob/master/ggml-easy.h
struct mimi_ggml_ctx {
    gguf_context * ctx_gguf = nullptr;
    ggml_context * ctx_data = nullptr;
    ggml_context * ctx_gf   = nullptr;

    // CPU-only for now, as many kernels are missing and we actually get less performance with GPU
    ggml_backend_t backend     = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_backend_sched_ptr sched;

    ggml_cgraph * gf = nullptr;
    std::vector<uint8_t> buf_compute_meta;
    int max_nodes = 16 * 1024;

    std::unordered_map<std::string, ggml_tensor *> tensors;

    mimi_ggml_ctx() {
        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        auto buft = ggml_backend_get_default_buffer_type(backend);
        sched.reset(
            ggml_backend_sched_new(&backend, &buft, 1, max_nodes, false)
        );
        buf_compute_meta.resize(max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());
    }

    void load_gguf(const char * fname) {
        ggml_context * meta = nullptr;

        gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_init_from_file(fname, params);

        // load tensors
        const int n_tensors = gguf_get_n_tensors(ctx_gguf);

        std::vector<uint8_t> read_buf;
        ggml_init_params ggml_params = {
            /*.mem_size   =*/ (n_tensors + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        };

        ctx_data = ggml_init(ggml_params);
        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            ggml_free(meta);
            throw std::runtime_error("cannot open model file for loading tensors");
        }

        // add tensors to context
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * t = ggml_get_tensor(meta, name);
            ggml_tensor * cur = ggml_dup_tensor(ctx_data, t);
            ggml_set_name(cur, name);
            tensors.insert({name, cur});
        }

        // alloc memory and offload data
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
        buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf, i);
            ggml_tensor * cur = ggml_get_tensor(ctx_data, name);
            const size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
            // printf("%s: Loading tensor \"%s\"\n", __func__, name);
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                ggml_free(meta);
                throw std::runtime_error(string_format("failed to seek for tensor: %s", name));
            }
            int num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buft_is_host(buft)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        printf("%s: Loaded %d tensors from %s\n", __func__, n_tensors, fname);
        fin.close();

        ggml_free(meta);
    }

    /**
     * Build a cgraph using the given builder function.
     *
     * The built cgraph will be stored in `ctx.gf`
     */
    void build_graph(std::function<void(ggml_context *, ggml_cgraph *)> builder_fn) {
        ggml_free(ctx_gf);
        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute_meta.size(),
            /*.mem_buffer =*/ buf_compute_meta.data(),
            /*.no_alloc   =*/ true,
        };

        ctx_gf = ggml_init(params);
        ggml_backend_sched_reset(sched.get());
        gf = ggml_new_graph_custom(ctx_gf, max_nodes, false);

        builder_fn(ctx_gf, gf);
        ggml_backend_sched_alloc_graph(sched.get(), gf);
    }

    ggml_status compute() {
        ggml_status status = ggml_backend_sched_graph_compute(sched.get(), gf);
        return status;
    }

    void set_tensor_data(const std::string & name, const void * data) {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
        }
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    }

    std::pair<ggml_tensor *, std::vector<uint8_t>> get_tensor_data(const std::string & name) {
        ggml_tensor * t = ggml_get_tensor(ctx_gf, name.c_str());
        if (!t) {
            throw std::runtime_error(string_format("tensor not found: %s", name.c_str()));
        }
        std::vector<uint8_t> data(ggml_nbytes(t));
        ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
        return std::make_pair(t, data);
    }

    ggml_tensor * get_weight(const char *fmt, ...) {
        std::vector<char> str(128);
        va_list va;
        va_start(va, fmt);
        vsnprintf(str.data(), 128, fmt, va);
        va_end(va);
        auto it = tensors.find(str.data());
        if (it == tensors.end()) {
            throw std::runtime_error(string_format("weight tensor not found: %s", str.data()));
        }
        return it->second;
    }

    ~mimi_ggml_ctx() {
        ggml_free(ctx_data);
        gguf_free(ctx_gguf);
        ggml_backend_buffer_free(buf);
    }
};

///////////////////////////////////////////////////////////////////////////
// extension to ggml.h
// TODO: add these ops to the library (ofc with a more optimized kernel)


// mode: (0) constant, (1) reflect, (2) replicate, (3) circular
// value is only used in "constant"
// only "constant" with 0.0f and "replicate" are implemented here
static ggml_tensor * ggml_pad_ext(ggml_context * ctx0, ggml_tensor * x, int mode,
        int64_t pad_left, int64_t pad_right, float value = 0.0f) {
    GGML_ASSERT(value == 0.0f); // we can technically use ggml_arange, but for simplication we only support 0.0f
    GGML_ASSERT(mode == 0 || mode == 2);
    if (pad_left > 0) {
        ggml_tensor * tmp = ggml_new_tensor_2d(ctx0, x->type, pad_left, x->ne[1]);
        if (mode == 0) {
            tmp = ggml_scale(ctx0, tmp, value);
        } else if (mode == 2) {
            ggml_tensor * elem = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], 0); // get first column
            tmp = ggml_repeat(ctx0, elem, tmp);
        }
        x = ggml_concat(ctx0, tmp, x, 0);
    }
    if (pad_right > 0) {
        ggml_tensor * tmp = ggml_new_tensor_2d(ctx0, x->type, pad_right, x->ne[1]);
        if (mode == 0) {
            tmp = ggml_scale(ctx0, tmp, value);
        } else if (mode == 2) {
            int64_t last = x->ne[0] - 1;
            ggml_tensor * elem = ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], last * ggml_element_size(x)); // get last column
            tmp = ggml_repeat(ctx0, elem, tmp);
        }
        x = ggml_concat(ctx0, x, tmp, 0);
    }
    return x;
}




///////////////////////////////////////////////////////////////////////////
// MimiConv and MimiConvTranspose

static int64_t div_ceil(int64_t a, int64_t b) {
    return a / b + (a % b ? 1 : 0);
}

static ggml_tensor * mimi_conv_1d(ggml_context * ctx0, ggml_tensor * x,
        ggml_tensor * kernel, ggml_tensor * bias, int stride, int dilation, bool pad_zero = true) {
    int64_t kernel_size = (kernel->ne[0] - 1) * dilation + 1;
    int64_t p_total = kernel_size - stride; // padding total
    int64_t p_half = p_total / 2;

    int64_t n_frames = div_ceil(x->ne[0] - kernel_size + p_total, stride);
    int64_t ideal_len = n_frames * stride + kernel_size - p_total;
    int64_t p_extra = ideal_len - x->ne[0];

    int64_t p_right = (mimi_config.causal ? 0 : p_half) + p_extra;
    int64_t p_left = p_total - (mimi_config.causal ? 0 : p_half);

    x = ggml_pad_ext(ctx0, x, pad_zero ? 0 : 2, p_left, p_right);

    x = ggml_conv_1d(ctx0, kernel, x, stride, 0, dilation);
    if (bias) {
        x = ggml_add(ctx0, x, bias);
    }
    ggml_set_name(x, "mimi_conv_1d");
    return x;
}

static ggml_tensor * mimi_conv_transpose_1d(ggml_context * ctx0, ggml_tensor * x,
        ggml_tensor * kernel, ggml_tensor * bias, int stride, int dilation, bool depthwise) {
    GGML_ASSERT(x->ne[1] == kernel->ne[2]);
    int64_t n_rows = x->ne[1];
    int64_t kernel_size = kernel->ne[0];
    int64_t p_total = kernel_size - stride; // padding total

    int64_t p_right = mimi_config.causal
        ? (float)p_total / mimi_config.trim_right_ratio
        : p_total / 2;
    int64_t p_left = p_total - p_right;

    ggml_tensor * out = nullptr;

    if (depthwise) {
        for (int64_t ir = 0; ir < n_rows; ir++) {
            ggml_tensor * row = ggml_view_1d(ctx0, x,
                                            x->ne[0], ir*x->ne[0]*ggml_element_size(x));
            ggml_tensor * krn = ggml_view_1d(ctx0, kernel,
                                            kernel->ne[0], ir*kernel->ne[0]*ggml_element_size(kernel));
            row = ggml_conv_transpose_1d(ctx0, krn, row, stride, 0, dilation);
            // unpad (remove p_right and p_left columns)
            row = ggml_view_1d(ctx0, row, row->ne[0] - p_total, p_left*ggml_element_size(row));

            // TODO: concat can be slow, we should use ggml_view_1d/ggml_cpy to avoid realloc
            out = out ? ggml_concat(ctx0, out, row, 1) : row;
        }

    } else {
        out = ggml_conv_transpose_1d(ctx0, kernel, x, stride, 0, dilation);
        // unpad
        out = ggml_view_2d(ctx0, out,
            out->ne[0] - p_total, out->ne[1],
            out->nb[1], p_left*ggml_element_size(out));
    }

    if (bias) {
        out = ggml_add(ctx0, out, bias);
    }

    return out;
}



///////////////////////////////////////////////////////////////////////////

// based on MimiEncoder
// SEANet encoder as used by Mimi.
struct mimi_encoder_decoder {
    mimi_ggml_ctx & ctx;
    struct layer {
        bool is_elu = false;
        bool is_resnet = false;
        bool is_transposed_conv = false;
        ggml_tensor * conv_0_w;
        ggml_tensor * conv_0_b;
        ggml_tensor * conv_1_w;
        ggml_tensor * conv_1_b;
        int stride = 1;
    };
    std::vector<layer> layers;

    std::array<int, 4> repeated_pattern = {1, 4, 7, 10};

    mimi_encoder_decoder(mimi_ggml_ctx & ctx): ctx(ctx) {
        layers.push_back({
            .conv_0_w = ctx.get_weight("decoder.layers.0.conv.weight"),
            .conv_0_b = ctx.get_weight("decoder.layers.0.conv.bias"),
        });
        for (int i = 0; i < (int)repeated_pattern.size(); ++i) {
            int i_start = repeated_pattern[i];
            // upsampling layers
            layers.push_back({
                .is_elu = true, // layer (i_start)
            });
            layers.push_back({
                .conv_0_w = ctx.get_weight("decoder.layers.%d.conv.weight", i_start + 1),
                .conv_0_b = ctx.get_weight("decoder.layers.%d.conv.bias",   i_start + 1),
                .stride = mimi_config.upsampling_ratio[i],
                .is_transposed_conv = true,
            });
            // residual layers
            layers.push_back({
                .is_resnet = true,
                .conv_0_w = ctx.get_weight("decoder.layers.%d.block.1.conv.weight", i_start + 2),
                .conv_0_b = ctx.get_weight("decoder.layers.%d.block.1.conv.bias",   i_start + 2),
                .conv_1_w = ctx.get_weight("decoder.layers.%d.block.3.conv.weight", i_start + 2),
                .conv_1_b = ctx.get_weight("decoder.layers.%d.block.3.conv.bias",   i_start + 2),
            });
        }
        layers.push_back({
            .is_elu = true, // layer 13
        });
        layers.push_back({
            .conv_0_w = ctx.get_weight("decoder.layers.14.conv.weight"),
            .conv_0_b = ctx.get_weight("decoder.layers.14.conv.bias"),
        });
    }

    ggml_tensor * forward(ggml_context * ctx0, ggml_tensor * input) {
        ggml_tensor * x = input;

        for (auto & layer : layers) {
            if (layer.is_elu) {
                x = ggml_elu(ctx0, x);
            } else if (layer.is_resnet) {
                ggml_tensor * residual = x;
                x = ggml_elu(ctx0, x);
                x = mimi_conv_1d(ctx0, x, layer.conv_0_w, layer.conv_0_b, 1, 1);
                x = ggml_elu(ctx0, x);
                x = mimi_conv_1d(ctx0, x, layer.conv_1_w, layer.conv_1_b, 1, 1);
                x = ggml_add(ctx0, x, residual);
            } else {
                x = layer.is_transposed_conv
                    ? mimi_conv_transpose_1d(ctx0, x, layer.conv_0_w, layer.conv_0_b, layer.stride, 1, false)
                    : mimi_conv_1d(ctx0, x, layer.conv_0_w, layer.conv_0_b, layer.stride, 1);
            }
        }

        return x;
    }
};

struct mimi_transformer {
    struct layer {
        ggml_tensor * inp_norm_w;
        ggml_tensor * inp_norm_b;

        ggml_tensor * attn_q;
        ggml_tensor * attn_k;
        ggml_tensor * attn_v;
        ggml_tensor * attn_o;
        ggml_tensor * attn_post_norm_w;
        ggml_tensor * attn_post_norm_b;
        ggml_tensor * attn_layer_scale;

        ggml_tensor * ffn_up;
        ggml_tensor * ffn_down;
        ggml_tensor * mlp_layer_scale;
    };
    std::vector<layer> layers;

    mimi_transformer(mimi_ggml_ctx & ctx, const char * prefix, int n_layers) {
        for (int il = 0; il < n_layers; il++) {
            layers.push_back({
                .inp_norm_w = ctx.get_weight("%s_transformer.layers.%d.input_layernorm.weight", prefix, il),
                .inp_norm_b = ctx.get_weight("%s_transformer.layers.%d.input_layernorm.bias",   prefix, il),

                .attn_q           = ctx.get_weight("%s_transformer.layers.%d.self_attn.q_proj.weight",         prefix, il),
                .attn_k           = ctx.get_weight("%s_transformer.layers.%d.self_attn.k_proj.weight",         prefix, il),
                .attn_v           = ctx.get_weight("%s_transformer.layers.%d.self_attn.v_proj.weight",         prefix, il),
                .attn_o           = ctx.get_weight("%s_transformer.layers.%d.self_attn.o_proj.weight",         prefix, il),
                .attn_post_norm_w = ctx.get_weight("%s_transformer.layers.%d.post_attention_layernorm.weight", prefix, il),
                .attn_post_norm_b = ctx.get_weight("%s_transformer.layers.%d.post_attention_layernorm.bias",   prefix, il),
                .attn_layer_scale = ctx.get_weight("%s_transformer.layers.%d.self_attn_layer_scale.scale",     prefix, il),

                .ffn_up          = ctx.get_weight("%s_transformer.layers.%d.mlp.fc1.weight",        prefix, il),
                .ffn_down        = ctx.get_weight("%s_transformer.layers.%d.mlp.fc2.weight",        prefix, il),
                .mlp_layer_scale = ctx.get_weight("%s_transformer.layers.%d.mlp_layer_scale.scale", prefix, il),
            });
        }
    }

    ggml_tensor * forward(ggml_context * ctx0, ggml_tensor * input, ggml_tensor * inp_pos) {
        int n_tokens    = input->ne[1];
        ggml_tensor * x = input;

        auto layer_norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
            x = ggml_norm(ctx0, x, mimi_config.norm_eps);
            x = ggml_mul(ctx0, x, w);
            x = ggml_add(ctx0, x, b);
            return x;
        };

        ggml_tensor * residual = input;

        for (auto & layer : layers) {
            residual = x;

            // input layer norm
            x = layer_norm(x, layer.inp_norm_w, layer.inp_norm_b);

            // self attention
            {
                ggml_tensor * q = ggml_mul_mat(ctx0, layer.attn_q, x);
                ggml_tensor * k = ggml_mul_mat(ctx0, layer.attn_k, x);
                ggml_tensor * v = ggml_mul_mat(ctx0, layer.attn_v, x);

                int n_embd_head = mimi_config.n_embd / mimi_config.n_head;
                q = ggml_reshape_3d(ctx0, q, n_embd_head, mimi_config.n_head,    n_tokens);
                k = ggml_reshape_3d(ctx0, k, n_embd_head, mimi_config.n_head_kv, n_tokens);
                v = ggml_reshape_3d(ctx0, v, n_embd_head, mimi_config.n_head_kv, n_tokens);

                int n_rot = n_embd_head;
                q = ggml_rope_inplace(ctx0, q, inp_pos, n_rot, 0);
                q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));

                k = ggml_rope_inplace(ctx0, k, inp_pos, n_rot, 0);
                k = ggml_cont(ctx0, ggml_permute(ctx0, k, 0, 2, 1, 3));

                ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
                ggml_mul_mat_set_prec(kq, GGML_PREC_F32); // mimic behavior of llama.cpp
                kq = ggml_scale_inplace(ctx0, kq, 1.0f / std::sqrt(n_embd_head));
                ggml_tensor * kq_masked = ggml_diag_mask_inf_inplace(ctx0, kq, n_tokens);
                kq = ggml_soft_max_inplace(ctx0, kq_masked);

                v = ggml_cont(ctx0, ggml_permute(ctx0, v, 1, 2, 0, 3));

                ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
                kqv = ggml_reshape_3d(ctx0, kqv, n_embd_head, n_tokens, mimi_config.n_head);
                kqv = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
                kqv = ggml_cont_2d(ctx0, kqv, mimi_config.n_embd, n_tokens);

                x = ggml_mul_mat(ctx0, layer.attn_o, kqv);
            }

            // residual
            x = ggml_mul(ctx0, x, layer.attn_layer_scale);
            x = ggml_add(ctx0, x, residual);

            residual = x;
            x = layer_norm(x, layer.attn_post_norm_w, layer.attn_post_norm_b);

            // mlp
            {
                x = ggml_mul_mat(ctx0, layer.ffn_up, x);
                x = ggml_gelu(ctx0, x);
                x = ggml_mul_mat(ctx0, layer.ffn_down, x);
            }

            // residual
            x = ggml_mul(ctx0, x, layer.mlp_layer_scale);
            x = ggml_add(ctx0, x, residual);
        }

        return x;
    }
};

struct mimi_residual_vector_quantizer {
    struct component {
        ggml_tensor * codebook;
    };

    ggml_tensor * semantic_inp_proj;
    std::vector<component> semantic_components;
    ggml_tensor * semantic_out_proj;

    ggml_tensor * acoustic_inp_proj;
    std::vector<component> acoustic_components;
    ggml_tensor * acoustic_out_proj;

    mimi_residual_vector_quantizer(mimi_ggml_ctx & ctx) {
        semantic_inp_proj = ctx.get_weight("quantizer.semantic_rvq.input_proj.weight");
        semantic_out_proj = ctx.get_weight("quantizer.semantic_rvq.output_proj.weight");
        for (int i = 0; i < mimi_config.n_semantic_components; i++) {
            semantic_components.push_back({
                .codebook = ctx.get_weight("quantizer.semantic_rvq.layers.%d.codebook",     i),
            });
        }
        acoustic_inp_proj = ctx.get_weight("quantizer.acoustic_rvq.input_proj.weight");
        acoustic_out_proj = ctx.get_weight("quantizer.acoustic_rvq.output_proj.weight");
        for (int i = 0; i < mimi_config.n_acoustic_components; i++) {
            acoustic_components.push_back({
                .codebook = ctx.get_weight("quantizer.acoustic_rvq.layers.%d.codebook",     i),
            });
        }
    }

    // the input has shape [n_codes, n_codes_per_embd]
    // first row is semantic, the rest are acoustic
    // example: [ [semantic], [acoustic1], [acoustic2], ... ]
    ggml_tensor * decode(ggml_context * ctx0, ggml_tensor * input) {
        GGML_ASSERT(input->type == GGML_TYPE_I32);

        size_t  n_semantic       = semantic_components.size();
        int64_t n_codes_per_embd = (n_semantic + acoustic_components.size());
        int64_t n_codes          = input->ne[0] / n_codes_per_embd;

        GGML_ASSERT(input->ne[0] % n_codes_per_embd == 0);

        ggml_tensor * out_s = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mimi_config.codebook_dim, n_codes);
        ggml_tensor * out_a = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, mimi_config.codebook_dim, n_codes);
        out_s = ggml_scale(ctx0, out_s, 0.0f); // clear
        out_a = ggml_scale(ctx0, out_a, 0.0f); // clear

        for (size_t ir = 0; ir < (size_t)n_codes_per_embd; ir++) {
            ggml_tensor * row = ggml_view_1d(ctx0, input, n_codes, ir*n_codes*ggml_element_size(input));
            if (ir < n_semantic) {
                // semantic
                ggml_tensor * codebook = semantic_components[ir].codebook;
                ggml_tensor * embd = ggml_get_rows(ctx0, codebook, row);
                out_s = ggml_add(ctx0, out_s, embd);
            } else {
                // acoustic
                ggml_tensor * codebook = acoustic_components[ir-n_semantic].codebook;
                ggml_tensor * embd = ggml_get_rows(ctx0, codebook, row);
                out_a = ggml_add(ctx0, out_a, embd);
            }
        }

        out_s = ggml_mul_mat(ctx0, semantic_out_proj, out_s);
        out_a = ggml_mul_mat(ctx0, acoustic_out_proj, out_a);

        return ggml_add(ctx0, out_s, out_a);
    }
};



///////////////////////////////////////////////////////////////////////////
// main program

int main(int argc, const char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf codes.txt [output.wav]\n", argv[0]);
        fprintf(stderr, "  Format of codes.txt file: one code per line\n");
        fprintf(stderr, "  Replace codes.txt with dummy0 and dummy1 for testing\n");
        fprintf(stderr, "    dummy0: using code 1, 2, 3,..., 96, used for logits matching\n");
        fprintf(stderr, "    dummy1: using code that will outputs 'hey hello there' sound\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * codes_path = argv[2];
    const char * out_path   = argc < 4 ? "output.wav" : argv[3];

    mimi_ggml_ctx ctx;
    ctx.load_gguf(model_path);

    // initialize components
    mimi_encoder_decoder           decoder(ctx);
    mimi_transformer               transformer(ctx, "decoder", mimi_config.num_hidden_layers);
    mimi_residual_vector_quantizer quantizer(ctx);

    // load codes
    std::vector<int> codes;
    if (strcmp(codes_path, "dummy0") == 0) {
        printf("Using dummy0 codes\n");
        codes.resize(32 * 3); // [n_codes = 3, n_codes_per_embd = 32]
        int n = 0;
        for (int c = 0; c < 32; c++) {
            for (int r = 0; r < 3; r++) {
                codes[r*32 + c] = n++;
            }
        }
    } else if (strcmp(codes_path, "dummy1") == 0) {
        printf("Using dummy1 codes\n");
        codes = {
            1263 ,1597 ,1596 ,1477 ,1540 ,1720 ,1433 ,118  ,1066 ,1968 ,1096 ,232  ,418  ,566  ,1653 ,2010 ,
            1029 ,1874 ,77   ,1803 ,123  ,908  ,97   ,1616 ,595  ,1170 ,1654 ,1211 ,1967 ,1579 ,1846 ,1462 ,
            1962 ,175  ,1539 ,742  ,1065 ,1226 ,19   ,955  ,528  ,1031 ,659  ,1687 ,1173 ,1802 ,1031 ,1714 ,
            1986 ,582  ,367  ,112  ,1245 ,1386 ,759  ,532  ,1472 ,1790 ,802  ,1213 ,1543 ,1916 ,1251 ,309  ,
            1962 ,1280 ,1943 ,878  ,1588 ,1989 ,568  ,1463 ,1814 ,1095 ,103  ,583  ,976  ,998  ,871  ,587  ,
            247  ,1698 ,1817 ,1024 ,268  ,597  ,45   ,1608 ,1880 ,2047 ,759  ,1578 ,1612 ,49   ,1031 ,1076 ,
            927  ,1202 ,1601 ,1719 ,1670 ,412  ,568  ,1838 ,341  ,1265 ,1279 ,830  ,1997 ,32   ,1369 ,1686 ,
            1307 ,419  ,1143 ,324  ,325  ,572  ,1597 ,1920 ,795  ,915  ,610  ,2000 ,819  ,718  ,1235 ,282  ,
            1912 ,1911 ,141  ,1069 ,1485 ,642  ,1370 ,732  ,284  ,1407 ,1591 ,1002 ,939  ,671  ,951  ,1411 ,
            1887 ,460  ,1588 ,1636 ,1312 ,232  ,969  ,1513 ,1336 ,1185 ,1660 ,4    ,926  ,1243 ,1077 ,1379 ,
            704  ,85   ,257  ,1302 ,1029 ,1717 ,899  ,1345 ,355  ,1915 ,1007 ,315  ,1283 ,779  ,415  ,335  ,
            1848 ,1786 ,469  ,295  ,380  ,1736 ,393  ,765  ,1921 ,836  ,374  ,1649 ,52   ,1633 ,759  ,548  ,
            1922 ,47   ,564  ,893  ,34   ,131  ,1063 ,1657 ,474  ,1960 ,1255 ,1275 ,92   ,976  ,1217 ,483  ,
            105  ,1746 ,1158 ,1557 ,1001 ,512  ,1668 ,1255 ,1045 ,1596 ,613  ,1272 ,1366 ,1147 ,411  ,831  ,
            349  ,692  ,1435 ,2005 ,1465 ,37   ,892  ,95   ,460  ,557  ,1315 ,259  ,1978 ,1838 ,1232 ,2003 ,
            1197 ,111  ,1953 ,1297 ,1843 ,671  ,1687 ,91   ,1788 ,1138 ,1896 ,399  ,615  ,758  ,1423 ,365  ,
            288  ,632  ,876  ,875  ,1156 ,345  ,1189 ,638  ,1527 ,1981 ,1925 ,333  ,1353 ,473  ,1913 ,1443 ,
            1634 ,1373 ,803  ,420  ,192  ,1440 ,1593 ,1925 ,784  ,831  ,552  ,807  ,1942 ,1289 ,612  ,511  ,
            968  ,1091 ,30   ,828  ,1611 ,1241 ,1985 ,596  ,273  ,529  ,1182 ,302  ,726  ,1942 ,733  ,1590 ,
            1564 ,214  ,1156 ,1722 ,1215 ,1837 ,1729 ,1823 ,672  ,116  ,340  ,396  ,721  ,462  ,1615 ,1380 ,
            1459 ,1553 ,636  ,586  ,1148 ,1147 ,1941 ,471  ,876  ,127  ,1938 ,2002 ,1563 ,1121 ,857  ,1179 ,
            1983 ,1324 ,1726 ,1445 ,295  ,270  ,896  ,1947 ,1740 ,1211 ,128  ,1266 ,734  ,715  ,1562 ,285  ,
            1139 ,304  ,526  ,653  ,1270 ,320  ,484  ,22   ,687  ,1065 ,489  ,827  ,993  ,1654 ,431  ,1552 ,
            1418 ,1604 ,455  ,841  ,412  ,848  ,475  ,540  ,1903 ,575  ,584  ,300  ,1079 ,189  ,1481 ,893  ,
            228  ,1577 ,429  ,635  ,106  ,1536 ,176  ,348  ,1733 ,1570 ,537  ,1840 ,798  ,410  ,1714 ,1318 ,
            487  ,332  ,1109 ,1744 ,283  ,692  ,681  ,1744 ,1008 ,1715 ,1956 ,1066 ,1768 ,1645 ,139  ,1967 ,
            897  ,132  ,1010 ,1932 ,277  ,1536 ,1541 ,952  ,19   ,88   ,1663 ,1232 ,1681 ,1878 ,1241 ,1805 ,
            89   ,1401 ,544  ,1061 ,1166 ,267  ,1351 ,1998 ,1623 ,1898 ,425  ,1320 ,2006 ,865  ,1981 ,823  ,
            1243 ,471  ,485  ,1765 ,391  ,1281 ,1607 ,1418 ,116  ,1702 ,1725 ,512  ,1088 ,1375 ,1994 ,1738 ,
            725  ,1471 ,811  ,1251 ,1156 ,1664 ,898  ,1511 ,1872 ,1717 ,444  ,1005 ,254  ,103  ,202  ,1769 ,
            1511 ,433  ,284  ,721  ,1741 ,56   ,615  ,916  ,887  ,1253 ,916  ,535  ,1666 ,1713 ,741  ,873  ,
            447  ,492  ,388  ,321  ,1860 ,1456 ,1658 ,1682 ,848  ,462  ,2034 ,1368 ,1609 ,1887 ,510  ,1516 ,
        };
    } else {
        std::ifstream fin(codes_path);
        if (!fin) {
            fprintf(stderr, "Error: cannot open codes file: %s\n", codes_path);
            return 1;
        }
        std::string line;
        while (std::getline(fin, line)) {
            // Skip empty lines
            if (line.empty()) continue;
            try {
                int code = std::stoi(line);
                codes.push_back(code);
            } catch (const std::exception& e) {
                fprintf(stderr, "Error parsing code: %s\n", line.c_str());
                return 1;
            }
        }
        if (codes.empty()) {
            fprintf(stderr, "Error: no codes found in file: %s\n", codes_path);
            return 1;
        }

        printf("Loaded %d codes from %s\n", (int)codes.size(), codes_path);
    }

    // build cgraph
    int n_pos            = -1;
    int n_codes          = codes.size();
    int n_codes_per_embd = mimi_config.n_semantic_components + mimi_config.n_acoustic_components;
    GGML_ASSERT(n_codes % n_codes_per_embd == 0 && "number of codes must be a multiple of n_codes_per_embd");

    ctx.build_graph([&](ggml_context * ctx_gf, ggml_cgraph * gf) {
        ggml_tensor * inp_dec = ggml_new_tensor_1d(ctx_gf, GGML_TYPE_I32, n_codes);
        ggml_set_name(inp_dec, "inp_dec");
        ggml_set_input(inp_dec);

        // RVQ
        ggml_tensor * embeddings = quantizer.decode(ctx_gf, inp_dec);

        // upsample
        embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
        embeddings = mimi_conv_transpose_1d(ctx_gf, embeddings, ctx.get_weight("upsample.conv.weight"), nullptr, 2, 1, true);

        // transformer
        n_pos = embeddings->ne[0];
        ggml_tensor * pos_dec = ggml_new_tensor_1d(ctx_gf, GGML_TYPE_I32, n_pos);
        ggml_set_name(pos_dec, "pos_dec");
        ggml_set_input(pos_dec);
        embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
        embeddings = transformer.forward(ctx_gf, embeddings, pos_dec);

        // SEANET decoder
        embeddings = ggml_cont(ctx_gf, ggml_transpose(ctx_gf, embeddings));
        ggml_tensor * output = decoder.forward(ctx_gf, embeddings);

        ggml_set_name(output, "output");
        ggml_set_output(output);
        ggml_build_forward_expand(gf, output);
    });

    // position data
    std::vector<int> pos_data(1024);
    for (int i = 0; i < (int)pos_data.size(); i++) {
        pos_data[i] = i;
    }
    ctx.set_tensor_data("pos_dec", pos_data.data());

    // code data (need to transpose it)
    // code [n_codes, n_codes_per_embd] -> [n_codes_per_embd, n_codes]
    std::vector<int> codes_t(n_codes_per_embd * n_codes);
    for (int i = 0; i < n_codes / n_codes_per_embd; i++) {
        for (int j = 0; j < n_codes_per_embd; j++) {
            int src_idx = i * n_codes_per_embd + j;
            int dst_idx = j * (n_codes / n_codes_per_embd) + i;
            codes_t[dst_idx] = codes[src_idx];
        }
    }
    ctx.set_tensor_data("inp_dec", codes_t.data());

    ctx.compute();

    auto output = ctx.get_tensor_data("output");
    auto output_tensor = output.first;
    auto output_data   = output.second;
    printf("Output shape: [%lld, %lld]\n", output_tensor->ne[0], output_tensor->ne[1]);

    // print first 20 values
    for (int i = 0; i < 20; i++) {
        printf("%2.4f, ", ((float *)output_data.data())[i]);
    }
    printf("...\n");

    // write to wav
    std::vector<float> wav_data(output_data.size() / sizeof(float));
    for (size_t i = 0; i < wav_data.size(); i++) {
        wav_data[i] = ((float *)output_data.data())[i];
    }
    printf("Writing to %s\n", out_path);
    save_wav16(out_path, wav_data, 24000);
}
