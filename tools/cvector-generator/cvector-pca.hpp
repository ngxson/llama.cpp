// PCA-based control vector: reduce the collected per-layer directions to a
// single principal component per layer (one vector of length n_embd).
//
// Input  (in_path) : GGUF with a single 3D tensor "directions" of shape
//                    [n_embd, n_row, n_layers] (produced by the collection step).
// Output (out_path): GGUF with, per layer il:
//                      direction.<il+1>   [n_embd]   (F32, 1D)
//                  i.e. the top-1 principal component of that layer's
//                  directions, with a deterministic sign so re-runs are stable.
//                  The tensor naming matches the existing control vector loader
//                  (common_control_vector_load_one), so the output is usable
//                  directly as a control vector file.
//
// Algorithm: for each layer we run power iteration on the Gram matrix in row
// space G = A^T A (shape [n_row, n_row], small) instead of the covariance
// A A^T (shape [n_embd, n_embd], which would blow up memory for large n_embd).
// The dominant eigenvector v of G is back-projected to embd space as
// u = A v (shape [n_embd]), then normalized and sign-fixed.
//
// Memory: each layer is computed in its own graph built on a 2D view of the 3D
// directions tensor, so only one layer's intermediates are alive at a time.
//
// Runs on GPU when available (Metal/CUDA), CPU otherwise. Ported/inlined here
// (no external dependency); the structure follows the existing pca.hpp but
// fixes its shape bug, runs all layers through the backend scheduler, and adds
// a deterministic sign convention.

#include "common.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cvector_pca {

struct pca_params {
    int   n_threads   = 1;
    int   n_iters     = 1000; // max power iterations per layer
    int   n_batch     = 20;   // iterations batched into one graph (more = faster, more memory)
    float tolerance   = 1e-6f;
};

// backends + scheduler, kept alive across the per-layer graphs
struct pca_ctx {
    ggml_backend_t       backend     = nullptr;
    ggml_backend_t       backend_cpu = nullptr;
    ggml_backend_sched_t sched       = nullptr;

    explicit pca_ctx(int /*n_threads*/) {}
    ~pca_ctx() {
        if (sched)       ggml_backend_sched_free(sched);
        if (backend_cpu) ggml_backend_free(backend_cpu);
        if (backend)     ggml_backend_free(backend);
    }

    void init() {
        LOG_INF("%s: initializing backends\n", __func__);
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        GGML_ASSERT(backend_cpu);

        backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend) {
            backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
        }
        if (!backend) {
            LOG_WRN("%s: no GPU backend, falling back to CPU\n", __func__);
            backend = backend_cpu;
        }
        LOG_INF("%s: compute backend = %s\n", __func__, ggml_backend_name(backend));

        ggml_backend_t backends[2] = { backend, backend_cpu };
        ggml_backend_buffer_type_t bufts[2] = {
            ggml_backend_get_default_buffer_type(backend),
            ggml_backend_get_default_buffer_type(backend_cpu),
        };
        sched = ggml_backend_sched_new(backends, bufts, backend == backend_cpu ? 1 : 2,
                                       GGML_DEFAULT_GRAPH_SIZE, false, true);
        GGML_ASSERT(sched);
    }
};

// Euclidean norm over ne[0] (broadcast across ne[1..3]).
static ggml_tensor * norm(ggml_context * ctx0, ggml_tensor * t) {
    return ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, t)));
}

// contiguous 2D slice of a 3D tensor along its 3rd dim (the layer dim). the
// slice shares storage with the base tensor, so the per-layer graph reads the
// already-uploaded directions buffer with no extra copy.
static ggml_tensor * view_2d_slice(ggml_context * ctx0, ggml_tensor * x, int idx) {
    GGML_ASSERT(idx < (int) x->ne[2]);
    return ggml_view_2d(ctx0, x, x->ne[0], x->ne[1],
            ggml_row_size(x->type, x->ne[0]),
            (size_t) idx * x->ne[0] * x->ne[1] * ggml_element_size(x));
}

// build the power-iteration graph for one layer:
//   A [n_embd, n_row] (2D view of the directions input)
//   G = A^T A         -> [n_row, n_row]
//   x = [n_row, 1]    (graph input, random start; carries the running eigenvector)
//   repeat n_batch: x <- G x ; x <- x / ||x||
//   out = A (x / ||x||) via mul_mat(cont(transpose(A)), x) -> [n_embd, 1]
//   out <- out / ||out||
// dist is the change ||x_new - x_old|| of the last iterated x, used to check
// convergence (the caller reads it and may re-seed x for more iterations).
static ggml_cgraph * build_pca_graph(ggml_context * ctx0, const pca_params & params,
        ggml_tensor * A, ggml_tensor * x,
        ggml_tensor *& out, ggml_tensor *& dist) {
    const size_t graph_size = GGML_DEFAULT_GRAPH_SIZE * 2;
    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, graph_size, false);

    // Gram matrix in row space: mul_mat(A, A) contracts ne[0]=n_embd,
    // result ne = [A->ne[1]=n_row, A->ne[1]=n_row] = [n_row, n_row].
    ggml_tensor * G = ggml_mul_mat(ctx0, A, A);
    ggml_set_name(G, "G");

    ggml_tensor * cur = x;
    ggml_tensor * last_dist = nullptr;
    for (int i = 0; i < params.n_batch; ++i) {
        ggml_tensor * y = ggml_mul_mat(ctx0, G, cur);          // y = G x   -> [n_row, 1]
        y = ggml_div(ctx0, y, norm(ctx0, y));                  // normalize
        ggml_format_name(y, "x_%d", i);

        // convergence measure: ||y - cur||
        ggml_tensor * diff = ggml_add(ctx0, y, ggml_scale(ctx0, cur, -1.0f));
        last_dist = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, diff)));
        ggml_format_name(last_dist, "dist_%d", i);

        cur = y;
        ggml_build_forward_expand(gf, last_dist);
    }

    // back-project to embd space: u = A v where v = cur (normalized). mul_mat
    // contracts ne[0], so we need A with ne[0]=n_row -> use cont(transpose(A)).
    // transpose(A) has ne = [n_row, n_embd]; cont makes it row-contiguous with
    // ne[0]=n_row, ne[1]=n_embd. mul_mat(AT, v=[n_row,1]) -> [n_embd, 1].
    ggml_tensor * AT = ggml_cont(ctx0, ggml_transpose(ctx0, A));
    ggml_tensor * u  = ggml_mul_mat(ctx0, AT, cur);
    u = ggml_div(ctx0, u, norm(ctx0, u));
    ggml_set_name(u, "out");

    out  = u;
    dist = last_dist;
    ggml_build_forward_expand(gf, out);
    return gf;
}

// load the 3D "directions" tensor from in_path into a host float buffer
// (shape [n_embd, n_row, n_layers]). also reads the controlvector.model_hint
// metadata so the output file can propagate which model these directions came
// from. returns false on error.
static bool load_directions(const std::string & in_path,
        std::vector<float> & data,
        int64_t & n_embd, int64_t & n_row, int64_t & n_layers,
        std::string & model_hint) {
    struct ggml_init_params params_meta = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx_meta = ggml_init(params_meta);

    struct gguf_init_params params_gguf = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &ctx_meta,
    };
    struct gguf_context * ctx_gguf = gguf_init_from_file(in_path.c_str(), params_gguf);
    if (!ctx_gguf) {
        LOG_ERR("%s: failed to open %s\n", __func__, in_path.c_str());
        ggml_free(ctx_meta);
        return false;
    }

    if (gguf_find_tensor(ctx_gguf, "directions") < 0) {
        LOG_ERR("%s: tensor 'directions' not found in %s\n", __func__, in_path.c_str());
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        return false;
    }

    ggml_tensor * directions = ggml_get_tensor(ctx_meta, "directions");
    GGML_ASSERT(directions && directions->type == GGML_TYPE_F32);
    GGML_ASSERT(directions->ne[3] == 1); // only up to 3 dims

    n_embd   = directions->ne[0];
    n_row    = directions->ne[1];
    n_layers = directions->ne[2];
    if (n_embd <= 0 || n_row <= 0 || n_layers <= 0) {
        LOG_ERR("%s: invalid directions shape [%" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                __func__, n_embd, n_row, n_layers);
        gguf_free(ctx_gguf);
        ggml_free(ctx_meta);
        return false;
    }

    data.resize((size_t) ggml_nelements(directions));
    memcpy(data.data(), directions->data, ggml_nbytes(directions));

    // propagate the model hint from the input (set during collection) to the
    // output, so the cvector file records which model it was built for.
    {
        int64_t kid = gguf_find_key(ctx_gguf, "controlvector.model_hint");
        if (kid >= 0 && gguf_get_kv_type(ctx_gguf, kid) == GGUF_TYPE_STRING) {
            model_hint = gguf_get_val_str(ctx_gguf, kid);
        } else {
            model_hint.clear();
            LOG_WRN("%s: controlvector.model_hint not found in %s\n", __func__, in_path.c_str());
        }
    }

    LOG_INF("%s: loaded directions [%" PRId64 ", %" PRId64 ", %" PRId64 "] from %s\n",
            __func__, n_embd, n_row, n_layers, in_path.c_str());

    gguf_free(ctx_gguf);
    ggml_free(ctx_meta);
    return true;
}

// write per-layer direction.<il+1> [n_embd] (1D, F32) to out_path. the tensor
// naming is 1-indexed to match the existing control vector loader. records the
// controlvector.{model_hint,layer_count} metadata propagated from the input.
static bool write_output(const std::string & out_path,
        const std::vector<std::vector<float>> & directions, int n_embd,
        const std::string & model_hint) {
    const int n_layers = (int) directions.size();
    const std::string arch = "controlvector";

    struct ggml_init_params params_ggml = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (size_t) (n_layers + 1),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_ggml = ggml_init(params_ggml);

    struct gguf_context * ctx_gguf = gguf_init_empty();
    gguf_set_val_str(ctx_gguf, "general.architecture", arch.c_str());
    if (!model_hint.empty()) {
        gguf_set_val_str(ctx_gguf, (arch + ".model_hint").c_str(), model_hint.c_str());
    }
    gguf_set_val_i32(ctx_gguf, (arch + ".layer_count").c_str(), n_layers);

    for (int il = 0; il < n_layers; ++il) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx_ggml, GGML_TYPE_F32, n_embd);
        ggml_format_name(t, "direction.%d", il + 1);
        t->data = (void *) directions[il].data();
        gguf_add_tensor(ctx_gguf, t);
    }

    LOG_INF("%s: writing %d layers (n_embd=%d) to %s\n",
            __func__, n_layers, n_embd, out_path.c_str());
    bool ok = gguf_write_to_file(ctx_gguf, out_path.c_str(), false);
    if (!ok) {
        LOG_ERR("%s: failed to write %s\n", __func__, out_path.c_str());
    }

    gguf_free(ctx_gguf);
    ggml_free(ctx_ggml);
    return ok;
}

// run PCA on one layer's 2D slice of the uploaded directions tensor. returns
// the principal component (length n_embd) with a deterministic sign. x is the
// working eigenvector [n_row]; the caller seeds it and re-feeds it across
// batches for continuity.
static bool pca_one_layer(pca_ctx & ctx, const pca_params & params,
        ggml_tensor * directions_3d, int il, int n_embd, int n_row,
        std::vector<float> & xv, std::vector<float> & out_vec) {
    // metadata pool for the graph + its (re-used) tensors. sized generously; the
    // graph holds ~4 ops per batched iteration plus the back-projection.
    const size_t graph_size = GGML_DEFAULT_GRAPH_SIZE * 2;
    struct ggml_init_params params_gf = {
        /*.mem_size   =*/ ggml_tensor_overhead() * graph_size + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx0 = ggml_init(params_gf);

    // 2D view of this layer's directions [n_embd, n_row]; shares the directions
    // buffer (which the caller already allocated on the compute backend), so no
    // set_input here - the view inherits the base tensor's backend buffer.
    ggml_tensor * A = view_2d_slice(ctx0, directions_3d, il);
    ggml_set_name(A, "A");

    // working eigenvector x [n_row, 1] (random start, seeded by caller)
    ggml_tensor * x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_row, 1);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_tensor * out = nullptr, * dist = nullptr;
    ggml_cgraph * gf = build_pca_graph(ctx0, params, A, x, out, dist);

    if (ggml_backend_is_cpu(ctx.backend)) {
        ggml_backend_cpu_set_n_threads(ctx.backend, params.n_threads);
    }

    auto get_tensor = [&](const char * name) {
        ggml_tensor * t = ggml_graph_get_tensor(gf, name);
        GGML_ASSERT(t && "tensor not found in graph");
        return t;
    };

    std::vector<float> out_buf((size_t) n_embd, 0.0f);
    float last_dist = 1e9f;
    int n_outer = std::max(1, params.n_iters / std::max(1, params.n_batch));

    for (int outer = 0; outer < n_outer; ++outer) {
        ggml_backend_sched_reset(ctx.sched);
        if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
            LOG_ERR("%s: layer %d: failed to allocate graph\n", __func__, il);
            ggml_free(ctx0);
            return false;
        }

        // once per layer: report scheduler splits and any op the compute backend
        // does not support (which would force a CPU fallback). n_splits > 1 means
        // the graph is split across backends.
        if (outer == 0) {
            int n_splits = ggml_backend_sched_get_n_splits(ctx.sched);
            LOG_INF("%s: layer %d: scheduler splits = %d, backend = %s\n",
                    __func__, il + 1, n_splits, ggml_backend_name(ctx.backend));
            int n_nodes = ggml_graph_n_nodes(gf);
            for (int i = 0; i < n_nodes; ++i) {
                ggml_tensor * node = ggml_graph_node(gf, i);
                if (node && !ggml_backend_supports_op(ctx.backend, node)) {
                    LOG_WRN("%s: layer %d: op '%s' (type %d) not supported by %s -> CPU\n",
                            __func__, il + 1, node->name, (int) node->op,
                            ggml_backend_name(ctx.backend));
                }
            }
        }

        // fetch the scheduler-wired tensors (the originals in ctx0 have no buffer)
        ggml_tensor * x_g = get_tensor("x");
        ggml_backend_tensor_set(x_g, xv.data(), 0, ggml_nbytes(x_g));

        ggml_status status = ggml_backend_sched_graph_compute(ctx.sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            LOG_ERR("%s: layer %d: graph compute failed (status %d)\n",
                    __func__, il, (int) status);
            ggml_free(ctx0);
            return false;
        }

        // read the final iterated x back as the next batch's seed, and the
        // convergence distance to decide whether we are done.
        {
            char xname[16];
            snprintf(xname, sizeof(xname), "x_%d", params.n_batch - 1);
            ggml_tensor * x_last = get_tensor(xname);
            ggml_backend_tensor_get(x_last, xv.data(), 0, ggml_nbytes(x_last));
        }
        {
            char dname[32];
            snprintf(dname, sizeof(dname), "dist_%d", params.n_batch - 1);
            ggml_tensor * d_last = get_tensor(dname);
            ggml_backend_tensor_get(d_last, &last_dist, 0, sizeof(float));
        }

        if (last_dist < params.tolerance) {
            break;
        }
    }

    // final graph already ran; read the normalized embd-space output
    ggml_tensor * out_g = get_tensor("out");
    ggml_backend_tensor_get(out_g, out_buf.data(), 0, ggml_nbytes(out_g));

    // deterministic sign: make the largest-magnitude component positive. power
    // iteration's eigenvector is only defined up to sign, so without this the
    // direction can flip across runs (the bug noted in the old pca.hpp).
    {
        int   best = 0;
        float best_abs = std::abs(out_buf[0]);
        for (int j = 1; j < n_embd; ++j) {
            float a = std::abs(out_buf[j]);
            if (a > best_abs) { best_abs = a; best = j; }
        }
        if (out_buf[best] < 0.0f) {
            for (int j = 0; j < n_embd; ++j) out_buf[j] = -out_buf[j];
        }
    }

    out_vec.swap(out_buf);
    LOG_INF("%s: layer %d done (last_dist=%.3e)\n", __func__, il + 1, last_dist);
    ggml_free(ctx0);
    return true;
}

static bool run(const std::string & in_path, const std::string & out_path, const pca_params & params) {
    LOG_INF("%s: step 1/4 init backends\n", __func__);
    pca_ctx ctx(params.n_threads);
    ctx.init();

    LOG_INF("%s: step 2/4 load directions from %s\n", __func__, in_path.c_str());
    std::vector<float> dir_data;
    int64_t n_embd = 0, n_row = 0, n_layers = 0;
    std::string model_hint;
    if (!load_directions(in_path, dir_data, n_embd, n_row, n_layers, model_hint)) {
        return false;
    }

    // graph-building context (metadata only; tensors get backend buffers from the
    // scheduler at alloc time, except directions_3d which we allocate ourselves).
    const size_t meta_size = ggml_tensor_overhead() * (GGML_DEFAULT_GRAPH_SIZE * 2 + 4)
                             + ggml_graph_overhead();
    struct ggml_init_params params_gf = {
        /*.mem_size   =*/ meta_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_gf = ggml_init(params_gf);

    // allocate the 3D directions tensor directly on the compute backend's buffer
    // (like clip.cpp does for weights), so the per-layer views stay on-device and
    // the scheduler runs the whole graph on GPU. doing this through the scheduler
    // instead leaves the input on the CPU buffer and forces a CPU fallback.
    // ctx_data must outlive the per-layer graphs (their views reference its
    // tensor via src[0]), so it is freed at the very end alongside dir_buf.
    struct ggml_init_params params_data = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2u,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_data = ggml_init(params_data);

    ggml_tensor * directions_3d = ggml_new_tensor_3d(ctx_data, GGML_TYPE_F32, n_embd, n_row, n_layers);
    ggml_set_name(directions_3d, "directions");

    ggml_backend_buffer_t dir_buf = nullptr;
    {
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(ctx.backend);
        dir_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
        if (!dir_buf && ctx.backend != ctx.backend_cpu) {
            // GPU buffer too small for the directions tensor: fall back to the
            // CPU buffer so the run still completes (the scheduler will copy
            // per-op). better than aborting.
            LOG_WRN("%s: directions buffer alloc failed on %s, falling back to CPU\n",
                    __func__, ggml_backend_name(ctx.backend));
            buft = ggml_backend_get_default_buffer_type(ctx.backend_cpu);
            dir_buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);
        }
    }
    if (!dir_buf) {
        LOG_ERR("%s: failed to allocate directions buffer on %s\n",
                __func__, ggml_backend_name(ctx.backend));
        ggml_free(ctx_data);
        ggml_free(ctx_gf);
        return false;
    }
    ggml_backend_buffer_set_usage(dir_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    LOG_INF("%s: uploaded directions (%.1f MB) to %s\n", __func__,
            (double) ggml_nbytes(directions_3d) / (1024.0 * 1024.0),
            ggml_backend_name(ctx.backend));
    ggml_backend_tensor_set(directions_3d, dir_data.data(), 0, ggml_nbytes(directions_3d));

    LOG_INF("%s: step 3/4 PCA over %d layers (n_embd=%lld, n_row=%lld, iters=%d, batch=%d)\n",
            __func__, (int) n_layers, (long long) n_embd, (long long) n_row,
            params.n_iters, params.n_batch);

    std::vector<std::vector<float>> out_layers(n_layers);
    std::mt19937 rng(0xc1ec5eed); // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int il = 0; il < (int) n_layers; ++il) {
        // random start eigenvector for this layer, normalized
        std::vector<float> xv((size_t) n_row);
        float ss = 0.0f;
        for (auto & f : xv) {
            f = dist01(rng);
            ss += f * f;
        }
        float inv = 1.0f / std::sqrt(ss);
        for (auto & f : xv) f *= inv;

        if (!pca_one_layer(ctx, params, directions_3d, il, (int) n_embd, (int) n_row, xv, out_layers[il])) {
            ggml_free(ctx_gf);
            ggml_backend_buffer_free(dir_buf);
            ggml_free(ctx_data);
            return false;
        }
    }

    LOG_INF("%s: step 4/4 write output to %s\n", __func__, out_path.c_str());
    bool ok = write_output(out_path, out_layers, (int) n_embd, model_hint);

    // free the per-layer graph metadata pool, then the directions device buffer
    // and its metadata context (must outlive the graphs, freed last).
    ggml_free(ctx_gf);
    ggml_backend_buffer_free(dir_buf);
    ggml_free(ctx_data);
    return ok;
}

} // namespace cvector_pca