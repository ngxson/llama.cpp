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
// Memory Layout Constants for Cross-Backend Consistency
// Block sizes are validated at compile time via static_assert in ggml-common.h:
//   static_assert(sizeof(block_q6_k_hifi_res8) == 232, ...)
//   static_assert(sizeof(block_q5_k_hifi_res8) == 200, ...)
// ===========================================================================

// Q6_K_HIFI_RES8: 232 bytes total (210 base + 22 extension)
// Layout: ql[128] + qh[64] + scales[16] + d[2] + outlier_count[1] + 
//         outlier_idx[8] + residual_vals[8] + _padding[1] + residual_scale[4]
#define Q6_K_HIFI_RES8_BLOCK_SIZE 232

// Q5_K_HIFI_RES8: 200 bytes total (176 base + 24 extension)
// Layout: dm[4] + scales[12] + qh[32] + qs[128] + outlier_count[1] +
//         outlier_idx[8] + residual_vals[8] + _padding[3] + residual_scale[4]
#define Q5_K_HIFI_RES8_BLOCK_SIZE 200

// ===========================================================================
// Q3_K_HIFI Adaptive Enhancement API
// Implements scale-aware tensor selection and statistical outlier detection
// ===========================================================================

// Q3_K_HIFI block constants
#ifndef Q3_K_HIFI_MAX_OUTLIERS
#define Q3_K_HIFI_MAX_OUTLIERS 8
#endif

// Model size categories for Q3_K_HIFI
typedef enum {
    Q3_HIFI_SIZE_TINY   = 0,  // ≤1.7B: minimal or no HIFI enhancement
    Q3_HIFI_SIZE_MEDIUM = 1,  // 2B-8B: full enhancement (sweet spot)
    Q3_HIFI_SIZE_LARGE  = 2,  // 14B+: reduced enhancement (leverage redundancy)
} ggml_q3_hifi_size_category;

// Get model size category from parameter count
// Parameters:
//   model_params_b: Model size in billions (e.g., 0.6, 1.7, 4.0, 8.0, 14.0, 32.0)
// Returns: Size category for adaptive strategy selection
GGML_API ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b);

// Get maximum outlier count for Q3_K_HIFI based on model size
// Implements Phase 1: Scale-Aware Enhancement
// Parameters:
//   model_params_b: Model size in billions
// Returns: Maximum outliers (0-8)
//   - Tiny (≤1.7B): 0-2 (avoid BPW overhead that hurts small models)
//   - Medium (2-8B): 6-8 (full enhancement - this is the sweet spot)
//   - Large (14B+): 3-4 (minimal enhancement - large models self-correct)
GGML_API int ggml_q3_hifi_get_max_outliers(float model_params_b);

// Get outlier ratio threshold for Q3_K_HIFI tensor enhancement
// Implements Phase 2: Statistical Outlier Detection
// Only enhance tensors whose outlier ratio exceeds this threshold
// Parameters:
//   model_params_b: Model size in billions
// Returns: Minimum outlier ratio (0.0-1.0) required for enhancement
//   - Tiny: 0.12 (12% - very selective to avoid wasting bits)
//   - Medium: 0.06 (6% - moderate selectivity)
//   - Large: 0.04 (4% - catch high-sensitivity tensors)
GGML_API float ggml_q3_hifi_get_outlier_threshold(float model_params_b);

// Compute statistical outlier ratio for a weight tensor
// Uses 3σ rule: count(|w| > 3 * stddev) / n_elements
// Parameters:
//   weights: Input weight tensor
//   n: Number of elements
// Returns: Outlier ratio (0.0-1.0)
GGML_API float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n);

// Determine if a tensor should receive Q3_K_HIFI enhancement
// Combines scale-aware and statistical outlier detection
// Parameters:
//   tensor_name: Name of the tensor (e.g., "blk.5.attn_v.weight")
//   weights: Weight data (can be NULL if only using name-based rules)
//   n_elements: Number of elements in tensor
//   model_params_b: Model size in billions
//   enhanced_count: Current count of enhanced tensors (in/out)
//   max_enhanced: Maximum tensors to enhance
// Returns: true if tensor should use HIFI enhancement
GGML_API int ggml_q3_hifi_should_enhance_tensor(
    const char * tensor_name,
    const float * weights,
    int64_t n_elements,
    float model_params_b,
    int * enhanced_count,
    int max_enhanced
);

// Get the enhancement type for Q3_K_HIFI critical tensors
// Parameters:
//   model_params_b: Model size in billions
//   is_embedding: Whether this is token_embd or output.weight
// Returns: GGML_TYPE to use (Q4_K, Q5_K, or Q6_K)
GGML_API int ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding);

// Get percentage of attn_v layers to enhance
// Parameters:
//   model_params_b: Model size in billions
// Returns: Threshold (0.0-1.0) - enhance layers where layer_idx <= n_layers * threshold
GGML_API float ggml_q3_hifi_get_attn_v_threshold(float model_params_b);

// ===========================================================================
// Q3_K_HIFI Per-Tensor Outlier Control (TLS)
// Allows dynamic outlier allocation per tensor based on imatrix importance
// ===========================================================================

// Set outlier count for the current tensor being quantized
// Pass -1 to use the default model-size-based count
// Parameters:
//   outliers: Outlier count (0-8) or -1 for default
GGML_API void ggml_q3_hifi_set_tensor_outliers(int outliers);

// Get the current tensor outlier count (-1 if using default)
// Returns: Outlier count or -1 if using default
GGML_API int ggml_q3_hifi_get_tensor_outliers(void);

// Set tensor importance for current quantization (from imatrix)
// Parameters:
//   importance: Importance score (0.0-1.0)
GGML_API void ggml_q3_hifi_set_tensor_importance(float importance);

// Get current tensor importance
// Returns: Importance score (0.0-1.0)
GGML_API float ggml_q3_hifi_get_tensor_importance(void);

// Reset TLS state to defaults (call after each tensor)
GGML_API void ggml_q3_hifi_reset_tensor_state(void);

// Compute adaptive outlier count for a specific block
// Used in per-block quantization for fine-grained control
// Parameters:
//   block_outlier_ratio: Outlier ratio for this specific block
//   base_outlier_count: Base outlier count from tensor-level decision
//   model_params_b: Model size in billions
// Returns: Adjusted outlier count for this block (0-8)
GGML_API int ggml_q3_hifi_compute_block_outliers(
    float block_outlier_ratio,
    int base_outlier_count,
    float model_params_b
);

#ifdef __cplusplus
}
#endif

#endif // GGML_QUANTS_HIFI_H

