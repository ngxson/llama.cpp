// GGML HIFI Quantization Context Implementation
// Layer-adaptive outlier allocation for Q4_HIFI quantization

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
    // Early layers (0-30%): Max precision - context formation is critical
    // Middle layers (30-70%): High precision - reasoning/processing (tuned up from 6)
    // Late layers (70-100%): Moderate precision - some redundancy in large models (tuned up from 4)
    int base_count;
    if (depth_ratio <= 0.30f) {
        base_count = 8;  // Early layers: max outliers
    } else if (depth_ratio <= 0.70f) {
        base_count = 7;  // Middle layers: high (tuned from 6)
    } else {
        base_count = 5;  // Late layers: moderate (tuned from 4)
    }

    // Scale-dependent adjustment
    // Key insight from testing:
    // - 8B models: Less aggressive reduction works better (more outliers in middle layers)
    // - 4B models: Aggressive reduction hurts quality - need similar approach to 8B
    // - Small models: Boost outliers everywhere
    float scale_factor = 1.0f;
    if (model_params_b >= 7.0f) {
        // 7B+ models: moderate reduction in middle and late layers
        if (depth_ratio > 0.70f) {
            scale_factor = 0.75f;  // Moderate reduction for late layers
        } else if (depth_ratio > 0.50f) {
            scale_factor = 0.9f;   // Slight reduction in middle-late
        }
    } else if (model_params_b >= 3.0f) {
        // 3-7B models (including 4B): Same approach as 8B - less aggressive
        // Testing showed aggressive reduction hurts quality at this scale
        if (depth_ratio > 0.70f) {
            scale_factor = 0.80f;  // Moderate late layer reduction (was 0.65)
        } else if (depth_ratio > 0.50f) {
            scale_factor = 0.95f;  // Very light middle-late reduction (was 0.80)
        }
        // Early and early-middle layers: no reduction (scale_factor = 1.0)
    } else if (model_params_b >= 1.5f) {
        // 1.5-3B models: light reduction in late layers only
        if (depth_ratio > 0.70f) {
            scale_factor = 0.85f;
        }
    } else if (model_params_b <= 1.0f) {
        // Small models (<1B): boost outliers everywhere
        // Small models are more sensitive to quantization error
        scale_factor = 1.2f;
        if (depth_ratio <= 0.30f) {
            scale_factor = 1.3f;  // Extra boost for early layers
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

