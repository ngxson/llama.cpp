// GGML HIFI Quantization Context
// Provides layer-adaptive outlier allocation for Q4_K_HIFI quantization
//
// This header defines the context infrastructure for passing layer-specific
// parameters to the quantization functions without modifying the core GGML API.

#ifndef GGML_QUANTS_HIFI_H
#define GGML_QUANTS_HIFI_H

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum outliers per block for Q6_K_HIFI_RES8 format
// Must match the value in ggml-common.h
#ifndef Q6_K_HIFI_RES8_MAX_OUTLIERS
#define Q6_K_HIFI_RES8_MAX_OUTLIERS 8
#endif

// Maximum outliers per block for Q5_K_HIFI_RES8 format
// Must match the value in ggml-common.h
#ifndef Q5_K_HIFI_RES8_MAX_OUTLIERS
#define Q5_K_HIFI_RES8_MAX_OUTLIERS 8
#endif

// Layer-adaptive quantization context
// Used to pass dynamic parameters to Q6_K_HIFI_RES8 quantization
typedef struct {
    int outlier_count;           // Number of outliers to preserve (1-8)
    float layer_importance;      // Layer importance score (0.0-1.0), for logging
    int layer_idx;               // Current layer index, for debugging
    int total_layers;            // Total layer count, for debugging
    int is_active;               // Whether adaptive mode is enabled
    float model_params_b;        // Model size in billions (e.g., 0.6, 1.7, 4.0, 8.0)
} ggml_hifi_quant_context;

// Get the current thread-local quantization context
// Returns NULL if no context is set
GGML_API const ggml_hifi_quant_context * ggml_hifi_get_context(void);

// Set the quantization context for the current thread
// Pass NULL to clear the context
GGML_API void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx);

// Convenience function to compute adaptive outlier count based on layer position and importance
// Parameters:
//   layer_idx: Current layer index (0-based)
//   total_layers: Total number of layers in the model
//   layer_importance: Normalized importance score (0.0-1.0), from imatrix aggregation
//   model_params_b: Model size in billions (e.g., 0.6, 1.7, 4.0, 8.0)
// Returns: Optimal outlier count (2-8)
GGML_API int ggml_hifi_compute_outlier_count(
    int layer_idx,
    int total_layers,
    float layer_importance,
    float model_params_b
);

// Convenience function to compute layer importance from imatrix data
// Parameters:
//   imatrix_data: Per-element importance weights from imatrix
//   n_elements: Number of elements in the tensor
// Returns: Aggregated importance score (0.0-1.0 after normalization)
GGML_API float ggml_hifi_compute_tensor_importance(
    const float * imatrix_data,
    int64_t n_elements
);

// Strategy 1: Compute per-block importance from imatrix data
// Used for adaptive per-block outlier allocation
// Parameters:
//   imatrix_block: Per-element importance weights for this block (QK_K elements)
//   block_size: Number of elements in the block (typically QK_K = 256)
// Returns: Block importance score (0.0-1.0)
GGML_API float ggml_hifi_compute_block_importance(
    const float * imatrix_block,
    int block_size
);

// Strategy 1: Compute per-block outlier count based on local imatrix variance
// High variance blocks get more outliers, low variance blocks get fewer
// Parameters:
//   block_importance: Importance score for this block (0.0-1.0)
//   base_outlier_count: Base outlier count from tensor-level computation
//   model_params_b: Model size in billions
// Returns: Adjusted outlier count for this block (2-8)
GGML_API int ggml_hifi_compute_block_outlier_count(
    float block_importance,
    int base_outlier_count,
    float model_params_b
);

// ===========================================================================
// Memory Layout Validators for Cross-Backend Consistency
// These macros validate block structure sizes and field offsets at compile time
// ===========================================================================

// Q6_K_HIFI_RES8 layout validation
// Total: 232 bytes (210 base + 22 extension)
#define Q6_K_HIFI_RES8_BLOCK_SIZE 232
#define Q6_K_HIFI_RES8_QL_OFFSET 0         // 128 bytes
#define Q6_K_HIFI_RES8_QH_OFFSET 128       // 64 bytes
#define Q6_K_HIFI_RES8_SCALES_OFFSET 192   // 16 bytes
#define Q6_K_HIFI_RES8_D_OFFSET 208        // 2 bytes (ggml_half)
#define Q6_K_HIFI_RES8_OUTLIER_COUNT_OFFSET 210  // 1 byte
#define Q6_K_HIFI_RES8_OUTLIER_IDX_OFFSET 211    // 8 bytes
#define Q6_K_HIFI_RES8_RESIDUAL_VALS_OFFSET 219  // 8 bytes
#define Q6_K_HIFI_RES8_PADDING_OFFSET 227        // 1 byte
#define Q6_K_HIFI_RES8_RESIDUAL_SCALE_OFFSET 228 // 4 bytes (float)

// Q5_K_HIFI_RES8 layout validation
// Total: 200 bytes (176 base + 24 extension)
#define Q5_K_HIFI_RES8_BLOCK_SIZE 200
#define Q5_K_HIFI_RES8_DM_OFFSET 0           // 4 bytes (2x ggml_half)
#define Q5_K_HIFI_RES8_SCALES_OFFSET 4       // 12 bytes (K_SCALE_SIZE)
#define Q5_K_HIFI_RES8_QH_OFFSET 16          // 32 bytes (QK_K/8)
#define Q5_K_HIFI_RES8_QS_OFFSET 48          // 128 bytes (QK_K/2)
#define Q5_K_HIFI_RES8_OUTLIER_COUNT_OFFSET 176  // 1 byte
#define Q5_K_HIFI_RES8_OUTLIER_IDX_OFFSET 177    // 8 bytes
#define Q5_K_HIFI_RES8_RESIDUAL_VALS_OFFSET 185  // 8 bytes
#define Q5_K_HIFI_RES8_PADDING_OFFSET 193        // 3 bytes
#define Q5_K_HIFI_RES8_RESIDUAL_SCALE_OFFSET 196 // 4 bytes (float)

// Runtime validation function - call during initialization
// Returns 0 on success, non-zero on layout mismatch
GGML_API int ggml_hifi_validate_memory_layout(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_QUANTS_HIFI_H

