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

// ===========================================================================
// Q3_K_HIFI Adaptive Enhancement Functions
// Implements scale-aware tensor selection and statistical outlier detection
// Based on proven strategies from Q4_K_HIFI and Q5_K_HIFI
// ===========================================================================

// Get model size category for Q3_K_HIFI adaptive strategy
ggml_q3_hifi_size_category ggml_q3_hifi_get_size_category(float model_params_b) {
    if (model_params_b <= 1.7f) {
        return Q3_HIFI_SIZE_TINY;   // 0.6B, 1.7B - minimal/no HIFI
    } else if (model_params_b <= 10.0f) {
        return Q3_HIFI_SIZE_MEDIUM; // 2B-8B - full HIFI (sweet spot)
    } else {
        return Q3_HIFI_SIZE_LARGE;  // 14B, 32B+ - reduced HIFI
    }
}

// Get maximum outlier count for Q3_K_HIFI based on model size
// Key insight from Q5_K_HIFI: Fixed enhancement doesn't scale!
// - Small models: HIFI overhead hurts more than it helps
// - Medium models: Full benefit from outlier preservation
// - Large models: Self-correcting, excessive outliers waste bits
int ggml_q3_hifi_get_max_outliers(float model_params_b) {
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);
    
    switch (cat) {
        case Q3_HIFI_SIZE_TINY:
            // ≤1.7B: 0-2 outliers
            // 0.6B especially struggles with BPW overhead
            if (model_params_b <= 0.8f) {
                return 0;  // Skip HIFI entirely for 0.6B
            }
            return 2;  // Minimal for 1.7B
            
        case Q3_HIFI_SIZE_MEDIUM:
            // 2B-8B: Full enhancement
            // This is where Q3_K_HIFI already wins (4B: -2.9% PPL)
            if (model_params_b <= 5.0f) {
                return 8;  // Max outliers for 2-5B
            }
            return 6;  // Slightly reduced for 8B
            
        case Q3_HIFI_SIZE_LARGE:
            // 14B+: Minimal enhancement
            // Large models have redundancy, extra outliers waste bits
            if (model_params_b >= 30.0f) {
                return 2;  // 32B+ gets minimal
            }
            return 4;  // 14B gets moderate
            
        default:
            return 4;  // Safe default
    }
}

// Get outlier ratio threshold for tensor enhancement decision
// Only enhance tensors with outlier ratio above this threshold
// Based on Q5_K_HIFI statistical detection patterns
float ggml_q3_hifi_get_outlier_threshold(float model_params_b) {
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);
    
    switch (cat) {
        case Q3_HIFI_SIZE_TINY:
            // Very selective - only enhance if absolutely needed
            return 0.12f;  // 12% threshold
            
        case Q3_HIFI_SIZE_MEDIUM:
            // Moderate selectivity - catch most high-sensitivity tensors
            if (model_params_b <= 5.0f) {
                return 0.06f;  // 6% for 2-5B
            }
            return 0.05f;  // 5% for 5-8B
            
        case Q3_HIFI_SIZE_LARGE:
            // Relaxed threshold - focus on highest-outlier tensors
            return 0.04f;  // 4% for 14B+
            
        default:
            return 0.08f;
    }
}

// Compute statistical outlier ratio using 3σ rule
// This is used to determine which tensors benefit from HIFI enhancement
float ggml_q3_hifi_compute_outlier_ratio(const float * weights, int64_t n) {
    if (weights == NULL || n <= 0) {
        return 0.0f;
    }
    
    // Single-pass mean and variance using Welford's algorithm
    double mean = 0.0;
    double m2 = 0.0;
    
    for (int64_t i = 0; i < n; ++i) {
        double x = (double)weights[i];
        double delta = x - mean;
        mean += delta / (double)(i + 1);
        double delta2 = x - mean;
        m2 += delta * delta2;
    }
    
    double variance = m2 / (double)n;
    if (variance <= 0.0) {
        return 0.0f;
    }
    
    double stddev = sqrt(variance);
    double threshold = 3.0 * stddev;
    
    // Count outliers (weights beyond 3σ from mean)
    int64_t outlier_count = 0;
    for (int64_t i = 0; i < n; ++i) {
        double diff = (double)weights[i] - mean;
        if (diff < 0) diff = -diff;  // fabs
        if (diff > threshold) {
            outlier_count++;
        }
    }
    
    return (float)outlier_count / (float)n;
}

// Determine if a tensor should receive Q3_K_HIFI enhancement
// Combines name-based rules, model size, and statistical analysis
int ggml_q3_hifi_should_enhance_tensor(
    const char * tensor_name,
    const float * weights,
    int64_t n_elements,
    float model_params_b,
    int * enhanced_count,
    int max_enhanced
) {
    if (enhanced_count == NULL) {
        return 0;
    }
    
    // Check if we've hit the enhancement limit
    if (*enhanced_count >= max_enhanced) {
        return 0;
    }
    
    // Always enhance critical tensors (if within budget)
    // token_embd and output.weight are always critical
    if (tensor_name != NULL) {
        // Check for critical path tensors
        const char * name = tensor_name;
        
        // token_embd.weight
        int is_token_embd = 0;
        const char * p = name;
        while (*p) {
            if (p[0] == 't' && p[1] == 'o' && p[2] == 'k' && p[3] == 'e' && p[4] == 'n' &&
                p[5] == '_' && p[6] == 'e' && p[7] == 'm' && p[8] == 'b' && p[9] == 'd') {
                is_token_embd = 1;
                break;
            }
            p++;
        }
        
        // output.weight
        int is_output = 0;
        p = name;
        while (*p) {
            if (p[0] == 'o' && p[1] == 'u' && p[2] == 't' && p[3] == 'p' && 
                p[4] == 'u' && p[5] == 't' && p[6] == '.') {
                is_output = 1;
                break;
            }
            p++;
        }
        
        if (is_token_embd || is_output) {
            (*enhanced_count)++;
            return 1;
        }
    }
    
    // For other tensors, use statistical outlier detection
    if (weights != NULL && n_elements > 0) {
        float outlier_ratio = ggml_q3_hifi_compute_outlier_ratio(weights, n_elements);
        float threshold = ggml_q3_hifi_get_outlier_threshold(model_params_b);
        
        if (outlier_ratio >= threshold) {
            (*enhanced_count)++;
            return 1;
        }
    }
    
    return 0;
}

// Get the enhancement type (Q4_K, Q5_K, or Q6_K) for critical tensors
// Returns GGML_TYPE_* values
int ggml_q3_hifi_get_enhancement_type(float model_params_b, int is_embedding) {
    // For Q3_K_HIFI, we use higher precision types for embeddings
    // Q6_K for embeddings (same as Q3_K_M default)
    // Q5_K for attn_v first layers (same as Q3_K_M)
    // Q4_K for other enhanced tensors
    
    if (is_embedding) {
        return 9;  // GGML_TYPE_Q6_K
    }
    
    // For large models, use higher precision on attn_v
    if (model_params_b >= 14.0f) {
        return 9;  // GGML_TYPE_Q6_K
    }
    
    // For medium models, Q5_K is a good balance
    if (model_params_b >= 4.0f) {
        return 8;  // GGML_TYPE_Q5_K
    }
    
    // For smaller models, Q4_K to avoid BPW overhead
    return 7;  // GGML_TYPE_Q4_K
}

// Get percentage of attn_v layers to enhance
// Based on model size - smaller models need broader coverage
// Aligned with llama-quant.cpp for consistency
float ggml_q3_hifi_get_attn_v_threshold(float model_params_b) {
    // Fine-grained thresholds matching llama-quant.cpp
    if (model_params_b <= 1.0f) {
        // 0.6B/1B: Skip attn_v HIFI entirely - matches Q3_K_M BPW
        // This addresses the +2.2% PPL regression seen at 0.6B
        return 0.0f;
    } else if (model_params_b <= 1.7f) {
        // 1.7B: Very minimal enhancement (2-3 layers only)
        return 0.07f;
    } else if (model_params_b <= 5.0f) {
        // 2-5B: Full enhancement - this is the sweet spot
        // 4B shows -2.9% PPL improvement with Q3_K_HIFI
        return 0.25f;
    } else if (model_params_b <= 10.0f) {
        // 5-8B: Moderate enhancement
        return 0.15f;
    } else if (model_params_b <= 20.0f) {
        // 14B: Reduced enhancement - addresses +0.24% PPL regression
        return 0.08f;
    } else {
        // 32B+: Minimal enhancement - addresses +0.13% PPL regression
        return 0.05f;
    }
}

// Compute adaptive outlier count for a specific block
// Fine-grained control based on per-block statistics
int ggml_q3_hifi_compute_block_outliers(
    float block_outlier_ratio,
    int base_outlier_count,
    float model_params_b
) {
    // If base count is 0, no outliers for this model size
    if (base_outlier_count <= 0) {
        return 0;
    }
    
    // Scale based on block's outlier ratio relative to tensor average
    // High ratio blocks get more outliers, low ratio blocks get fewer
    float threshold = ggml_q3_hifi_get_outlier_threshold(model_params_b);
    
    float scale = 1.0f;
    if (block_outlier_ratio >= threshold * 2.0f) {
        // Very high outlier block - boost significantly
        scale = 1.5f;
    } else if (block_outlier_ratio >= threshold) {
        // Above threshold - slight boost
        scale = 1.2f;
    } else if (block_outlier_ratio < threshold * 0.5f) {
        // Well below threshold - reduce
        scale = 0.6f;
    } else {
        // Near threshold - keep base
        scale = 0.9f;
    }
    
    // Model size adjustment
    ggml_q3_hifi_size_category cat = ggml_q3_hifi_get_size_category(model_params_b);
    if (cat == Q3_HIFI_SIZE_LARGE) {
        // Large models: more aggressive reduction
        scale *= 0.8f;
    } else if (cat == Q3_HIFI_SIZE_TINY) {
        // Tiny models: if we're using outliers at all, be conservative
        scale *= 1.2f;
    }
    
    int result = (int)roundf((float)base_outlier_count * scale);
    
    // Clamp to valid range
    if (result < 0) result = 0;
    if (result > Q3_K_HIFI_MAX_OUTLIERS) result = Q3_K_HIFI_MAX_OUTLIERS;
    
    return result;
}

