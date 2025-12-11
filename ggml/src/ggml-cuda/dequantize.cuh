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

static __device__ __forceinline__ void dequantize_q3_hifi(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q3_hifi * x = (const block_q3_hifi *) vx;

    const float d = __half2float(x[ib].d);
    const uint8_t * qs = x[ib].qs;

    // Extract two 3-bit values starting at iqs
    // Each value is 3 bits, so we need to unpack from the packed format
    int idx0 = iqs;
    int idx1 = iqs + 1;

    // Extract first value
    const int byte_idx0 = (idx0 * 3) / 8;
    const int bit_offset0 = (idx0 * 3) % 8;
    uint8_t bits0 = (qs[byte_idx0] >> bit_offset0) & 7;
    if (bit_offset0 > 5 && byte_idx0 + 1 < 96) {
        bits0 |= (qs[byte_idx0 + 1] << (8 - bit_offset0)) & 7;
    }
    const int quant_val0 = (int)bits0 - 4; // [0,7] → [-4,3]

    // Extract second value
    const int byte_idx1 = (idx1 * 3) / 8;
    const int bit_offset1 = (idx1 * 3) % 8;
    uint8_t bits1 = (qs[byte_idx1] >> bit_offset1) & 7;
    if (bit_offset1 > 5 && byte_idx1 + 1 < 96) {
        bits1 |= (qs[byte_idx1 + 1] << (8 - bit_offset1)) & 7;
    }
    const int quant_val1 = (int)bits1 - 4; // [0,7] → [-4,3]

    v.x = quant_val0 * d;
    v.y = quant_val1 * d;

    // Check if either index is an outlier and restore if so
    for (int k = 0; k < Q3_HIFI_OUTFIERS_PER_BLOCK; ++k) {
        if (x[ib].outlier_idx[k] == idx0) {
            v.x = __half2float(x[ib].outlier_vals[k]);
        }
        if (x[ib].outlier_idx[k] == idx1) {
            v.y = __half2float(x[ib].outlier_vals[k]);
        }
    }
}
