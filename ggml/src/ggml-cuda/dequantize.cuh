#include "common.cuh"

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// Q3_HIFI: Q3_K-compatible layout with 6 FP16 outliers
// Uses same hmask/qs/scales layout as Q3_K for the first 110 bytes
static __device__ __forceinline__ void dequantize_q3_hifi(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q3_hifi * x = (const block_q3_hifi *) vx;

    // Use Q3_K-style extraction
    const float d = __half2float(x[ib].d);
    const uint8_t * qs = x[ib].qs;
    const uint8_t * hmask = x[ib].hmask;
    
    // iqs is in range [0, QK_K/2) = [0, 128)
    // We need to extract 2 values at positions iqs*2 and iqs*2+1
    int idx0 = iqs * 2;
    int idx1 = iqs * 2 + 1;
    
    // Q3_K bit layout:
    // - qs[64]: lower 2 bits packed as 4 values per byte
    // - hmask[32]: high bit packed as 8 values per byte
    
    // Extract first value
    const int qs_byte0 = idx0 / 4;
    const int qs_shift0 = (idx0 % 4) * 2;
    const int hm_byte0 = idx0 / 8;
    const int hm_shift0 = idx0 % 8;
    const int lo0 = (qs[qs_byte0] >> qs_shift0) & 0x03;
    const int hi0 = (hmask[hm_byte0] >> hm_shift0) & 0x01;
    int quant_val0 = (lo0 | (hi0 << 2)) - 4;
    
    // Extract second value
    const int qs_byte1 = idx1 / 4;
    const int qs_shift1 = (idx1 % 4) * 2;
    const int hm_byte1 = idx1 / 8;
    const int hm_shift1 = idx1 % 8;
    const int lo1 = (qs[qs_byte1] >> qs_shift1) & 0x03;
    const int hi1 = (hmask[hm_byte1] >> hm_shift1) & 0x01;
    int quant_val1 = (lo1 | (hi1 << 2)) - 4;
    
    v.x = quant_val0 * d;
    v.y = quant_val1 * d;

    // Check if either index is an outlier and restore if so
    // Outliers are sparse (only 8 per 256 weights), so this loop is cheap
    #pragma unroll
    for (int k = 0; k < Q3_HIFI_OUTLIERS; ++k) {
        if (x[ib].outlier_idx[k] == idx0) {
            v.x = __half2float(x[ib].outlier_vals[k]);
        }
        if (x[ib].outlier_idx[k] == idx1) {
            v.y = __half2float(x[ib].outlier_vals[k]);
        }
    }
}
