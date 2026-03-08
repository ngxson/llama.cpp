#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

// Copy Q5_K base (176 bytes) from each Q5_K_HIFI_RES8 block (196 bytes) for MMQ path.
// Uses vectorized 4-byte loads: 176/4=44 words, 196/4=49 words (both divisible by 4 so every
// block-start is uint32_t-aligned regardless of block index).
static_assert(sizeof(block_q5_K)           % sizeof(uint32_t) == 0, "Q5_K size not a multiple of 4");
static_assert(sizeof(block_q5_k_hifi_res8) % sizeof(uint32_t) == 0, "Q5_K_HIFI_RES8 size not a multiple of 4");
static __global__ void ggml_cuda_compact_q5_k_hifi_res8_to_q5_k(
    const void * __restrict__ src, void * __restrict__ dst, int64_t n_blocks) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) return;
    const uint32_t * s = (const uint32_t *)((const char *)src + i * sizeof(block_q5_k_hifi_res8));
    uint32_t       * d = (uint32_t       *)((char       *)dst + i * sizeof(block_q5_K));
    #pragma unroll
    for (int j = 0; j < (int)(sizeof(block_q5_K) / sizeof(uint32_t)); ++j) {
        d[j] = s[j];
    }
}

// Add Q5_K_HIFI_RES8 INT8 residual corrections to MMQ output using F32 activations.
// Parallelised at the (row, block) level rather than (row, batch):
//   - 92% of threads hit the early-exit (outlier_count==0) before touching src1 or dst.
//   - The 8% of threads that do have outliers loop over all batch slots and atomicAdd
//     their contribution.  Contention is negligible (~1 writer per output cell on average).
static __global__ void ggml_cuda_add_q5_k_hifi_res8_residuals(
    const block_q5_k_hifi_res8 * __restrict__ x,
    const float * __restrict__ src1, float * __restrict__ dst,
    int64_t nrows_x, int64_t ncols_x, int64_t ncols_dst,
    int64_t stride_row_x, int64_t stride_src1, int64_t stride_dst) {

    const int64_t n_blocks = ncols_x / QK_K;
    const int64_t rb = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (rb >= nrows_x * n_blocks) return;

    const int64_t row = rb / n_blocks;
    const int64_t b   = rb % n_blocks;

    const block_q5_k_hifi_res8 * block = x + row * stride_row_x + b;
    const int n_out = (block->outlier_count & 0x7F);
    if (n_out == 0) return;           // fast path: ~92% of blocks exit here

    const uint8_t e4m3 = block->residual_scale_e4m3;
    if (e4m3 == 0) return;

    // Decode E4M3 FP8 residual scale once, in registers
    const int   sign     = (e4m3 >> 7) & 0x01;
    const int   exp      = (e4m3 >> 3) & 0x0F;
    const int   mantissa =  e4m3       & 0x07;
    const float res_scale = (1.0f + (float)mantissa * 0.125f)
                          * exp2f((float)exp - 7.0f)
                          * (sign ? -1.0f : 1.0f)
                          * (1.0f / 127.0f);

    // Cache per-outlier column indices and scaled residual values in registers
    // so the inner batch loop only reads src1 (no repeated block struct accesses).
    const int n_valid = (n_out < Q5_K_HIFI_RES8_MAX_OUTLIERS) ? n_out : Q5_K_HIFI_RES8_MAX_OUTLIERS;
    int   cols [Q5_K_HIFI_RES8_MAX_OUTLIERS];
    float rvals[Q5_K_HIFI_RES8_MAX_OUTLIERS];
    for (int k = 0; k < n_valid; ++k) {
        cols [k] = (int)b * QK_K + block->outlier_idx[k];
        rvals[k] = res_scale * (float)block->residual_vals[k];
    }

    // Accumulate residual dot-products over all batch slots and atomicAdd to dst.
    // Low contention: at most ~1.3 enhanced blocks per row on average.
    for (int64_t batch = 0; batch < ncols_dst; ++batch) {
        float sum = 0.0f;
        for (int k = 0; k < n_valid; ++k) {
            sum += rvals[k] * src1[batch * stride_src1 + cols[k]];
        }
        atomicAdd(&dst[batch * stride_dst + row], sum);
    }
}

// K_TURBO compact-copy kernels: strip residual extension, produce base-type blocks for MMQ.
// All TURBO types have base fields at identical byte offsets as the base type.
// Note: Q3_K = 110 bytes (not 4-aligned), so we use byte-by-byte copy to handle all cases.
static_assert(sizeof(block_q2_K)         % sizeof(uint32_t) == 0, "Q2_K size not a multiple of 4");
static_assert(sizeof(block_q2_k_turbo)   % sizeof(uint32_t) == 0, "Q2_K_TURBO size not a multiple of 4");
static_assert(sizeof(block_q3_k_turbo)   % sizeof(uint32_t) == 0, "Q3_K_TURBO size not a multiple of 4");
static_assert(sizeof(block_q4_K)         % sizeof(uint32_t) == 0, "Q4_K size not a multiple of 4");
static_assert(sizeof(block_q4_k_turbo)   % sizeof(uint32_t) == 0, "Q4_K_TURBO size not a multiple of 4");
static_assert(sizeof(block_q5_k_turbo)   % sizeof(uint32_t) == 0, "Q5_K_TURBO size not a multiple of 4");
static_assert(sizeof(block_q6_k_turbo)   % sizeof(uint32_t) == 0, "Q6_K_TURBO size not a multiple of 4");

#define DEFINE_COMPACT_TURBO_KERNEL(TNAME, TURBO_T, BASE_T) \
static __global__ void ggml_cuda_compact_##TNAME##_to_base( \
    const void * __restrict__ src, void * __restrict__ dst, int64_t n_blocks) { \
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= n_blocks) return; \
    const uint8_t * s = (const uint8_t *)((const char *)src + i * sizeof(TURBO_T)); \
    uint8_t       * d = (uint8_t       *)((char       *)dst + i * sizeof(BASE_T)); \
    _Pragma("unroll") \
    for (int j = 0; j < (int)sizeof(BASE_T); ++j) { d[j] = s[j]; } \
}

DEFINE_COMPACT_TURBO_KERNEL(Q2_K_TURBO, block_q2_k_turbo, block_q2_K)
DEFINE_COMPACT_TURBO_KERNEL(Q3_K_TURBO, block_q3_k_turbo, block_q2_K)  // Q3_K_TURBO base = Q2_K
DEFINE_COMPACT_TURBO_KERNEL(Q4_K_TURBO, block_q4_k_turbo, block_q3_K)  // Q4_K_TURBO base = Q3_K (110 bytes)
DEFINE_COMPACT_TURBO_KERNEL(Q5_K_TURBO, block_q5_k_turbo, block_q4_K)  // Q5_K_TURBO base = Q4_K
DEFINE_COMPACT_TURBO_KERNEL(Q6_K_TURBO, block_q6_k_turbo, block_q5_K)  // Q6_K_TURBO base = Q5_K

// Generic TURBO residual correction kernel.
// TURBO residual_scale = max_err / 127.0f (pre-divided), so correction = rscale * residual_vals[k].
// Launches one thread per (weight-row, block) pair; loops over batch dimension inside.
template<typename TURBO_T, int MAX_RESIDUALS>
static __global__ void ggml_cuda_add_turbo_residuals(
    const TURBO_T * __restrict__ x,
    const float * __restrict__ src1, float * __restrict__ dst,
    int64_t nrows_x, int64_t ncols_x, int64_t ncols_dst,
    int64_t stride_row_x, int64_t stride_src1, int64_t stride_dst) {

    const int64_t n_blocks = ncols_x / QK_K;
    const int64_t rb = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (rb >= nrows_x * n_blocks) return;

    const int64_t row = rb / n_blocks;
    const int64_t b   = rb % n_blocks;

    const TURBO_T * block = x + row * stride_row_x + b;
    const int rc = block->residual_count;
    if (rc == 0) return;  // fast path: most blocks have no residuals

    const float rscale = __half2float(block->residual_scale);
    const int n_valid = (rc < MAX_RESIDUALS) ? rc : MAX_RESIDUALS;

    // Cache per-residual column indices and scaled values in registers
    int   cols [MAX_RESIDUALS];
    float rvals[MAX_RESIDUALS];
    for (int k = 0; k < n_valid; ++k) {
        cols [k] = (int)b * QK_K + block->residual_idx[k];
        rvals[k] = rscale * (float)block->residual_vals[k];
    }

    // Accumulate over all batch slots
    for (int64_t batch = 0; batch < ncols_dst; ++batch) {
        float sum = 0.0f;
        for (int k = 0; k < n_valid; ++k) {
            sum += rvals[k] * src1[batch * stride_src1 + cols[k]];
        }
        atomicAdd(&dst[batch * stride_dst + row], sum);
    }
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream) {
    switch (args.type_x) {
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    // TODO: tighter pool buffer size vs q8 path
    const bool use_native_mxfp4 = blackwell_mma_available(cc) && src0->type == GGML_TYPE_MXFP4;

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_mxfp4) {
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_mxfp4_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_mxfp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) /
                                    (8 * QK_MXFP4 * sizeof(int))  // block_fp4_mmq holds 256 values (8 blocks of 32)
                                :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        if (src0->type == GGML_TYPE_Q5_K_HIFI_RES8) {
            const int64_t n_blocks = (ne00 / QK_K) * ne01;
            ggml_cuda_pool_alloc<char> q5_k_compact(ctx.pool(), n_blocks * sizeof(block_q5_K));
            const int nth = 256;
            ggml_cuda_compact_q5_k_hifi_res8_to_q5_k<<<(n_blocks + nth - 1) / nth, nth, 0, stream>>>
                (src0_d, q5_k_compact.get(), n_blocks);
            CUDA_CHECK(cudaGetLastError());
            const mmq_args args_q5 = {
                q5_k_compact.get(), GGML_TYPE_Q5_K, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
                ne00, ne01, ne1, s01, ne11, s1,
                ne02, ne12, s02, s12, s2,
                ne03, ne13, s03, s13, s3,
                use_stream_k, ne1};
            ggml_cuda_mul_mat_q_switch_type(ctx, args_q5, stream);
            const int64_t stride_src1 = src1->nb[1] / (int64_t)sizeof(float);
            const int64_t stride_dst  = dst->nb[1]  / (int64_t)sizeof(float);
            // Launch one thread per (weight-row, block) pair.
            // ~92% of threads exit immediately (no outliers); only ~8% touch src1/dst.
            const int64_t n_blocks_per_row = ne00 / QK_K;
            const int64_t n_rb = ne01 * n_blocks_per_row;
            ggml_cuda_add_q5_k_hifi_res8_residuals<<<(n_rb + 255) / 256, 256, 0, stream>>>
                ((const block_q5_k_hifi_res8 *)src0_d, (const float *)src1_d, dst_d,
                 ne01, ne00, ne1, s01, stride_src1, stride_dst);
            CUDA_CHECK(cudaGetLastError());
            return;
        }

#define TURBO_MMQ_PATH(TNAME, TURBO_T, BASE_SIZE, BASE_GGML_TYPE, MAX_RES) \
        if (src0->type == GGML_TYPE_##TNAME) { \
            const int64_t n_blocks = (ne00 / QK_K) * ne01; \
            ggml_cuda_pool_alloc<char> base_compact(ctx.pool(), n_blocks * BASE_SIZE); \
            const int nth = 256; \
            ggml_cuda_compact_##TNAME##_to_base<<<(n_blocks + nth - 1) / nth, nth, 0, stream>>>( \
                src0_d, base_compact.get(), n_blocks); \
            CUDA_CHECK(cudaGetLastError()); \
            const mmq_args args_base = { \
                base_compact.get(), BASE_GGML_TYPE, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d, \
                ne00, ne01, ne1, s01, ne11, s1, \
                ne02, ne12, s02, s12, s2, \
                ne03, ne13, s03, s13, s3, \
                use_stream_k, ne1}; \
            ggml_cuda_mul_mat_q_switch_type(ctx, args_base, stream); \
            const int64_t stride_src1 = src1->nb[1] / (int64_t)sizeof(float); \
            const int64_t stride_dst  = dst->nb[1]  / (int64_t)sizeof(float); \
            const int64_t n_blocks_per_row = ne00 / QK_K; \
            const int64_t n_rb = ne01 * n_blocks_per_row; \
            ggml_cuda_add_turbo_residuals<TURBO_T, MAX_RES><<<(n_rb + 255) / 256, 256, 0, stream>>>( \
                (const TURBO_T *)src0_d, (const float *)src1_d, dst_d, \
                ne01, ne00, ne1, s01, stride_src1, stride_dst); \
            CUDA_CHECK(cudaGetLastError()); \
            return; \
        }

        TURBO_MMQ_PATH(Q2_K_TURBO, block_q2_k_turbo, sizeof(block_q2_K), GGML_TYPE_Q2_K, Q2_K_TURBO_MAX_RESIDUALS)
        TURBO_MMQ_PATH(Q3_K_TURBO, block_q3_k_turbo, sizeof(block_q2_K), GGML_TYPE_Q2_K, Q3_K_TURBO_MAX_RESIDUALS)  // base = Q2_K
        TURBO_MMQ_PATH(Q4_K_TURBO, block_q4_k_turbo, sizeof(block_q3_K), GGML_TYPE_Q3_K, Q4_K_TURBO_MAX_RESIDUALS)  // base = Q3_K
        TURBO_MMQ_PATH(Q5_K_TURBO, block_q5_k_turbo, sizeof(block_q4_K), GGML_TYPE_Q4_K, Q5_K_TURBO_MAX_RESIDUALS)  // base = Q4_K
        TURBO_MMQ_PATH(Q6_K_TURBO, block_q6_k_turbo, sizeof(block_q5_K), GGML_TYPE_Q5_K, Q6_K_TURBO_MAX_RESIDUALS)  // base = Q5_K

        const mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            use_stream_k, ne1};
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * sizeof(block_q8_1)/QK8_1 +
        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_mxfp4) {
            quantize_mmq_mxfp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                    ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = use_native_mxfp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (8 * QK_MXFP4 * sizeof(int)) :
                                           ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    const mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        use_stream_k, ne12};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
}

void ggml_cuda_op_mul_mat_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;
    const int64_t stride01 = ne00 / ggml_blck_size(src0->type);

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    // The stream-k decomposition is only faster for recent NVIDIA GPUs.
    // Also its fixup needs to allocate a temporary buffer in the memory pool.
    // There are multiple parallel CUDA streams for src1_ncols != ne11 which would introduce a race condition for this buffer.
    const bool use_stream_k = ((GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc))
                            && src1_ncols == ne11;
    const mmq_args args = {
        src0_dd_i, src0->type, (const int *) src1_ddq_i, nullptr, nullptr, dst_dd_i,
        ne00, row_diff, src1_ncols, stride01, ne11, nrows_dst,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        use_stream_k, src1_ncols};

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_padded_row_size);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        // Q2_K_HIFI excluded - uses MMVQ/dequant path instead
        // Q3_K_HIFI excluded - uses MMVQ/dequant path instead
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q5_K_HIFI_RES8:  // Use Q5_K MMQ path (compact copy + residual kernel)
        case GGML_TYPE_Q2_K_TURBO:  // compact copy to Q2_K + residual correction
        case GGML_TYPE_Q3_K_TURBO:  // compact copy to Q2_K + residual correction (base shifted down)
        case GGML_TYPE_Q4_K_TURBO:  // compact copy to Q3_K + residual correction (base shifted down)
        case GGML_TYPE_Q5_K_TURBO:  // compact copy to Q4_K + residual correction (base shifted down)
        case GGML_TYPE_Q6_K_TURBO:  // compact copy to Q5_K + residual correction (base shifted down)
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        return false;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;

}
