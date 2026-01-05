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

// Maximum outliers per block for Q5_K_HIFI_HYBRID format
// Must match the values in ggml-common.h
#ifndef Q5_K_HIFI_HYBRID_MAX_EXTREME
#define Q5_K_HIFI_HYBRID_MAX_EXTREME 4   // Max FP16 outliers
#endif
#ifndef Q5_K_HIFI_HYBRID_MAX_MODERATE
#define Q5_K_HIFI_HYBRID_MAX_MODERATE 3  // Max INT8 outliers
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

#ifdef __cplusplus
}
#endif

#endif // GGML_QUANTS_HIFI_H

