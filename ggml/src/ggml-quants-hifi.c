// GGML HIFI Quantization Context Implementation
// Layer-adaptive outlier allocation for Q4_K_HIFI quantization

#include "ggml-quants-hifi.h"
#include <math.h>
#include <stdlib.h>

// Thread-local storage for the quantization context
// Using a simple pointer approach - the context lifetime is managed by the caller
#ifdef _MSC_VER
    static __declspec(thread) const ggml_hifi_quant_context * g_hifi_context = NULL;
#else
    static __thread const ggml_hifi_quant_context * g_hifi_context = NULL;
#endif

const ggml_hifi_quant_context * ggml_hifi_get_context(void) {
    return g_hifi_context;
}

void ggml_hifi_set_context(const ggml_hifi_quant_context * ctx) {
    g_hifi_context = ctx;
}

// Compute adaptive outlier count based on layer position, importance, and model scale
// This is the core algorithm for layer-wise imatrix adaptation
// Strategy 2 optimization: More aggressive reduction in middle/late layers
int ggml_hifi_compute_outlier_count(
    int layer_idx,
    int total_layers,
    float layer_importance,
    float model_params_b
) {
    if (total_layers <= 0) {
        return 8; // Default to max for safety
    }

    // Compute depth ratio (0.0 = first layer, 1.0 = last layer)
    float depth_ratio = (float)layer_idx / (float)(total_layers - 1);
    if (total_layers == 1) depth_ratio = 0.5f;

    // Base outlier count based on layer position
    // Strategy 2: More aggressive reduction for size optimization
    // Early layers (0-30%): Max precision - context formation is critical
    // Middle layers (30-70%): Reduced precision (5 instead of 7)
    // Late layers (70-100%): Minimal precision (2 instead of 5)
    int base_count;
    if (depth_ratio <= 0.30f) {
        base_count = 8;  // Early layers: max outliers (unchanged)
    } else if (depth_ratio <= 0.70f) {
        base_count = 5;  // Middle layers: reduced (was 7)
    } else {
        base_count = 2;  // Late layers: minimal (was 5)
    }

    // Scale-dependent adjustment
    // Key insight: Large models have more redundancy, can use fewer outliers
    // Small models need more outliers to maintain quality
    float scale_factor = 1.0f;
    if (model_params_b >= 7.0f) {
        // 7B+ models: already minimal late layers, no further reduction needed
        // But we can slightly reduce middle layers for extra savings
        if (depth_ratio > 0.30f && depth_ratio <= 0.70f) {
            scale_factor = 0.9f;  // Middle layers: slight reduction
        }
    } else if (model_params_b >= 3.0f) {
        // 3-7B models: Moderate approach
        if (depth_ratio > 0.70f) {
            scale_factor = 1.0f;  // Late layers already at minimum
        } else if (depth_ratio > 0.30f) {
            scale_factor = 0.95f; // Middle layers: very light reduction
        }
    } else if (model_params_b >= 1.5f) {
        // 1.5-3B models: Be more conservative, boost late layers slightly
        if (depth_ratio > 0.70f) {
            scale_factor = 1.25f; // Boost late layers back up (2 -> ~3)
        }
    } else if (model_params_b <= 1.0f) {
        // Small models (<1B): boost outliers everywhere
        // Small models are more sensitive to quantization error
        scale_factor = 1.3f;
        if (depth_ratio <= 0.30f) {
            scale_factor = 1.4f;  // Extra boost for early layers
        } else if (depth_ratio > 0.70f) {
            scale_factor = 1.5f;  // Late layers need more for small models (2 -> 3)
        }
    }

    // Apply importance adjustment
    // layer_importance is normalized to 0.0-1.0
    // High importance (>0.7): boost outlier count
    // Low importance (<0.3): reduce outlier count
    float importance_factor = 1.0f;
    if (layer_importance > 0.7f) {
        importance_factor = 1.0f + (layer_importance - 0.7f);  // Up to 1.3x
    } else if (layer_importance < 0.3f) {
        importance_factor = 0.7f + (layer_importance / 0.3f) * 0.3f;  // 0.7-1.0x
    }

    // Combine factors
    float final_count_f = (float)base_count * scale_factor * importance_factor;
    int final_count = (int)roundf(final_count_f);

    // Clamp to valid range [2, 8]
    if (final_count < 2) final_count = 2;
    if (final_count > 8) final_count = 8;

    return final_count;
}

// Compute tensor importance from imatrix data
// Uses the average of squared importance weights as the metric
float ggml_hifi_compute_tensor_importance(
    const float * imatrix_data,
    int64_t n_elements
) {
    if (imatrix_data == NULL || n_elements <= 0) {
        return 0.5f;  // Default to medium importance if no data
    }

    // Compute mean squared importance
    // This weights larger importance values more heavily
    double sum_sq = 0.0;
    double sum = 0.0;
    for (int64_t i = 0; i < n_elements; ++i) {
        double val = (double)imatrix_data[i];
        sum += val;
        sum_sq += val * val;
    }

    // Use coefficient of variation as importance metric
    // High variance in importance = some weights are critical = high importance
    double mean = sum / (double)n_elements;
    double mean_sq = sum_sq / (double)n_elements;
    double variance = mean_sq - mean * mean;

    if (mean < 1e-10 || variance < 0) {
        return 0.5f;
    }

    // Coefficient of variation (CV) = stddev / mean
    double stddev = sqrt(variance);
    double cv = stddev / mean;

    // Normalize CV to 0-1 range
    // Empirically, CV values typically range from 0.1 to 3.0 for imatrix data
    // Map this to 0.2 - 0.9 importance range
    float importance = 0.2f + 0.7f * (float)(cv / 3.0);
    if (importance > 0.9f) importance = 0.9f;
    if (importance < 0.2f) importance = 0.2f;

    return importance;
}

// Strategy 1: Compute per-block importance from imatrix data
// Uses coefficient of variation within the block as the importance metric
float ggml_hifi_compute_block_importance(
    const float * imatrix_block,
    int block_size
) {
    if (imatrix_block == NULL || block_size <= 0) {
        return 0.5f;  // Default to medium importance
    }

    // Compute statistics for this block
    double sum = 0.0;
    double sum_sq = 0.0;
    double max_val = 0.0;
    
    for (int i = 0; i < block_size; ++i) {
        double val = (double)imatrix_block[i];
        sum += val;
        sum_sq += val * val;
        if (val > max_val) max_val = val;
    }

    double mean = sum / (double)block_size;
    if (mean < 1e-10) {
        return 0.3f;  // Low importance for near-zero blocks
    }

    double mean_sq = sum_sq / (double)block_size;
    double variance = mean_sq - mean * mean;
    if (variance < 0) variance = 0;

    // Coefficient of variation (CV)
    double stddev = sqrt(variance);
    double cv = stddev / mean;

    // Also consider the max/mean ratio (spikiness)
    double spikiness = max_val / mean;

    // Combine CV and spikiness for final importance
    // High CV = high variance = some weights are outliers = need more outliers
    // High spikiness = extreme values present = need more outliers
    double combined = 0.6 * cv + 0.4 * (spikiness / 10.0);  // spikiness typically 1-20
    
    // Normalize to 0.2 - 0.9 range
    float importance = 0.2f + 0.7f * (float)(combined / 2.0);  // combined typically 0-3
    if (importance > 0.9f) importance = 0.9f;
    if (importance < 0.2f) importance = 0.2f;

    return importance;
}

// Strategy 1: Compute per-block outlier count based on local imatrix variance
// Adjusts the base outlier count up or down based on block importance
int ggml_hifi_compute_block_outlier_count(
    float block_importance,
    int base_outlier_count,
    float model_params_b
) {
    // Scale factor based on block importance
    // High importance (>0.7): boost outliers up to 1.5x
    // Low importance (<0.3): reduce outliers down to 0.5x
    // Medium importance: keep base count
    float scale = 1.0f;
    
    if (block_importance > 0.7f) {
        // High importance block - boost outliers
        scale = 1.0f + 0.5f * (block_importance - 0.7f) / 0.3f;  // 1.0 to 1.5
    } else if (block_importance < 0.3f) {
        // Low importance block - reduce outliers
        scale = 0.5f + 0.5f * (block_importance / 0.3f);  // 0.5 to 1.0
    }
    
    // For larger models, be more aggressive with reduction on low-importance blocks
    if (model_params_b >= 7.0f && block_importance < 0.4f) {
        scale *= 0.8f;  // Additional 20% reduction for large models
    }
    
    int adjusted_count = (int)roundf((float)base_outlier_count * scale);
    
    // Clamp to valid range [1, 8]
    // Allow minimum of 1 for low-importance blocks (save more space)
    if (adjusted_count < 1) adjusted_count = 1;
    if (adjusted_count > 8) adjusted_count = 8;
    
    return adjusted_count;
}

