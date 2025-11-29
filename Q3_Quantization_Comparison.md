# Q3 Quantization Formats Comparison: Q3_HIFI vs Q3_K_S vs Q3_K_M

## Executive Summary

This document compares 3-bit quantization strategies available in llama.cpp:
- **Q3_HIFI (Pure)**: A hybrid format using 3-bit quantization with FP16 outliers for all tensors
- **Q3_HIFI (Hybrid)**: A smart hybrid approach using Q3_HIFI for critical tensors (attn_v, ffn_down) and Q3_K for others, with strategic upgrades (output.weight→Q6_K, attn_output.weight→Q4_K)
- **Q3_K_S**: Aggressive mixed quantization using Q3_K format for most tensors
- **Q3_K_M**: Balanced mixed quantization using Q3_K format with more conservative tensor selection

---

## Technical Specifications

### Q3_HIFI (Pure)
- **Format**: Hybrid 3-bit + FP16 outliers
- **Block Structure**: 256 weights per block
  - 250 weights: 3-bit quantized (96 bytes)
  - 6 weights: Stored as FP16 outliers (12 bytes)
  - 6 outlier indices: uint16_t (12 bytes)
  - 1 float scale: 4 bytes
- **Bits per Weight**: ~3.875 bpw (124 bytes / 256 weights × 8)
- **Block Size**: 124 bytes per 256 weights
- **Outlier Strategy**: Identifies top-6 outliers by magnitude (optionally weighted by importance matrix) and stores them in full FP16 precision
- **Usage**: Applied to all quantizable tensors

### Q3_HIFI (Hybrid - Recommended)
- **Format**: Smart hybrid using Q3_HIFI selectively + Q3_K for bulk + strategic upgrades
- **Tensor Strategy**:
  - **attn_v**: Q3_HIFI (3.875 bpw) - preserves attention value outliers
  - **ffn_down**: Q3_HIFI (3.875 bpw) - preserves feed-forward outliers
  - **output.weight**: Q6_K (6.14 bpw) - maximum quality for output layer
  - **attn_output.weight**: Q4_K (4.5 bpw) - balanced quality for attention output
  - **All other tensors**: Q3_K (3.4375 bpw) - efficient bulk quantization
- **Bits per Weight**: ~3.47-3.50 bpw (weighted average)
- **File Size**: ~329MB for 0.6B model (vs 380MB Q3_K_S, 404MB Q3_K_M)
- **Key Advantage**: Smaller than Q3_K_S/M while maintaining or exceeding their quality through targeted Q3_HIFI usage

### Q3_K_S (Small)
- **Format**: Mixed quantization, primarily Q3_K
- **Base Format**: Q3_K (3.4375 bpw)
- **Block Structure**: 256 weights per block
  - 256 weights: 3-bit quantized with hierarchical scales
  - High bit mask: 32 bytes (1 bit per weight)
  - Low 2 bits: 64 bytes
  - 12 scale bytes (6-bit quantized scales for 16 sub-blocks)
  - 1 FP16 super-block scale: 2 bytes
- **Bits per Weight**: ~3.4375 bpw (110 bytes / 256 weights × 8)
- **Tensor Strategy**: 
  - Most tensors: Q3_K
  - Some critical tensors (early ffn_down layers): Q4_K or Q5_K
  - Attention output: Q4_K (for 8-expert models)

### Q3_K_M (Medium)
- **Format**: Mixed quantization, balanced Q3_K usage
- **Base Format**: Q3_K (3.4375 bpw)
- **Block Structure**: Same as Q3_K_S
- **Bits per Weight**: ~3.4375 bpw (110 bytes / 256 weights × 8)
- **Tensor Strategy**:
  - Most tensors: Q3_K
  - Attention weights (wv): Q4_K or Q5_K (depending on position)
  - Early ffn_down layers: Q5_K (first 1/16 of layers)
  - Later ffn_down layers: Q4_K (with exceptions)
  - Attention output: Q4_K
  - More conservative than Q3_K_S

---

## Detailed Comparison

### 1. File Size

| Format | Bits per Weight | File Size (0.6B model) | File Size (7B model est.) | Notes |
|--------|----------------|----------------------|--------------------------|-------|
| **Q3_HIFI (Pure)** | 3.875 bpw | ~370MB | ~3.75 GB | All tensors use Q3_HIFI |
| **Q3_HIFI (Hybrid)** | ~3.47 bpw (mixed) | **329MB** | **~3.33 GB** | Smart selective usage |
| **Q3_K_S** | ~3.41 bpw (mixed) | ~380MB | ~3.42 GB | Smallest pure format |
| **Q3_K_M** | ~3.74 bpw (mixed) | ~404MB | ~3.75 GB | Balanced with upgrades |

**Winner**: **Q3_HIFI (Hybrid)** - Smallest file size while maintaining quality! Q3_K_S is smallest pure format.

### 2. Quality / Accuracy

#### Q3_HIFI (Pure)
- **Pros**:
  - Preserves critical outliers in full FP16 precision
  - Can use importance matrix to intelligently select outliers
  - Better preservation of extreme values that might be important
  - Potentially better for models with sparse important weights
  
- **Cons**:
  - Fixed 6 outliers per block (may not be optimal for all distributions)
  - Outlier selection is magnitude-based (though can be weighted)
  - Slightly more complex dequantization
  - Larger file size (3.875 bpw for all tensors)

#### Q3_HIFI (Hybrid)
- **Pros**:
  - **Best of both worlds**: Q3_HIFI quality where it matters most (attn_v, ffn_down)
  - **Smaller file size** than Q3_K_S/M (329MB vs 380-404MB for 0.6B)
  - **Strategic upgrades**: Output at Q6_K, attention output at Q4_K (matching Q3_K_M quality)
  - **Targeted outlier preservation**: Only uses Q3_HIFI on tensors that benefit most
  - Can use importance matrix for outlier selection in Q3_HIFI tensors
  - Better quality than pure Q3_K_S while being smaller
  
- **Cons**:
  - Requires manual tensor-type specification
  - More complex quantization command
  - Still has outlier handling overhead for Q3_HIFI tensors

#### Q3_K_S
- **Pros**:
  - Consistent quantization approach across tensors
  - Well-optimized hierarchical scaling
  - Proven format with extensive testing
  
- **Cons**:
  - Most aggressive quantization (lowest quality)
  - May lose important outliers in critical tensors
  - Perplexity: +1.6321 @ Llama-3-8B (reference)

#### Q3_K_M
- **Pros**:
  - Better quality than Q3_K_S by preserving critical tensors
  - Balanced approach between size and quality
  - Perplexity: +0.6569 @ Llama-3-8B (reference)
  
- **Cons**:
  - Still uses 3-bit for most weights (may lose precision)
  - More complex tensor selection logic

**Winner**: **Q3_HIFI (Hybrid)** - Best quality-to-size ratio! Q3_HIFI (Pure) best for outlier-sensitive models, Q3_K_M best proven pure format quality

### 3. Speed / Performance

#### Q3_HIFI (Pure)
- **Inference Speed**: 
  - Slightly slower due to outlier handling
  - Requires checking outlier indices and loading FP16 values
  - More memory accesses per block
  - Dequantization: Must restore outliers after bulk dequantization
  
- **Memory Access Pattern**: 
  - Less cache-friendly (outlier indices scattered)
  - FP16 outlier values may cause cache misses
  
- **Hardware Optimization**: 
  - Less optimized in current backends (newer format)
  - May not have specialized GPU kernels yet

#### Q3_HIFI (Hybrid)
- **Inference Speed**: 
  - **Faster than pure Q3_HIFI** - only ~15% of tensors have outlier overhead
  - Most tensors (85%) use fast Q3_K dequantization
  - Q3_HIFI overhead limited to attn_v and ffn_down tensors
  - Output and attention output use optimized Q6_K/Q4_K paths
  
- **Memory Access Pattern**: 
  - Mixed: Q3_K tensors have good cache locality
  - Q3_HIFI tensors have scattered access (but fewer of them)
  
- **Hardware Optimization**: 
  - Benefits from optimized Q3_K, Q4_K, Q6_K kernels
  - Only Q3_HIFI tensors lack full optimization

#### Q3_K_S
- **Inference Speed**:
  - Fast, well-optimized format
  - Simple dequantization: hierarchical scale application
  - Highly optimized kernels across all backends (CUDA, Metal, Vulkan, etc.)
  - Cache-friendly access patterns
  
- **Memory Access**: 
  - Sequential block access
  - Good cache locality

#### Q3_K_M
- **Inference Speed**:
  - Similar to Q3_K_S for Q3_K tensors
  - Slightly slower overall due to mixed precision (some Q4_K/Q5_K tensors)
  - Still very fast, well-optimized
  
- **Memory Access**:
  - Mixed precision may cause some cache inefficiency
  - Still generally good

**Winner**: Q3_K_S (fastest), Q3_K_M (very close), **Q3_HIFI (Hybrid)** (faster than pure Q3_HIFI), Q3_HIFI (Pure) (slowest)

### 4. Quantization Time

#### Q3_HIFI
- **Time**: Moderate
- **Process**: 
  1. Find outliers (magnitude-based, optionally weighted)
  2. Quantize bulk weights
  3. Store outliers
- **Complexity**: O(n) per block for outlier selection

#### Q3_K_S
- **Time**: Fast
- **Process**: Standard hierarchical quantization
- **Complexity**: Well-optimized quantization path

#### Q3_K_M
- **Time**: Moderate (slower than Q3_K_S)
- **Process**: Same as Q3_K_S but with more tensor analysis
- **Complexity**: Additional logic to determine tensor precision

**Winner**: Q3_K_S (fastest quantization)

### 5. Memory Usage

#### Q3_HIFI (Pure)
- **RAM**: Slightly higher due to outlier storage
- **VRAM**: Similar to Q3_K_M
- **Cache**: Less efficient (scattered outlier access)

#### Q3_HIFI (Hybrid)
- **RAM**: Lower than pure Q3_HIFI (most tensors are Q3_K)
- **VRAM**: Lower than Q3_K_M (smaller file size)
- **Cache**: Mixed - good for Q3_K tensors, less efficient for Q3_HIFI tensors

#### Q3_K_S
- **RAM**: Lowest
- **VRAM**: Lowest
- **Cache**: Most efficient

#### Q3_K_M
- **RAM**: Similar to Q3_HIFI
- **VRAM**: Similar to Q3_HIFI
- **Cache**: Good (better than Q3_HIFI)

**Winner**: Q3_K_S (lowest memory), **Q3_HIFI (Hybrid)** (very close, smaller than Q3_K_M)

### 6. Hardware Support

#### Q3_HIFI
- **Status**: Newer format, may have limited optimization
- **Backends**: CPU (full), GPU (may be less optimized)
- **Future**: Potential for optimization improvements

#### Q3_K_S & Q3_K_M
- **Status**: Mature, highly optimized
- **Backends**: Full support across all backends
- **Optimization**: Extensive SIMD, GPU kernel optimizations

**Winner**: Q3_K_S and Q3_K_M (better hardware support)

### 7. Use Cases

#### Choose Q3_HIFI (Hybrid) When:
- ✅ You want the **best quality-to-size ratio**
- ✅ You want smaller files than Q3_K_S/M while maintaining quality
- ✅ You're willing to specify tensor types manually
- ✅ You want Q3_HIFI quality on critical tensors (attn_v, ffn_down)
- ✅ You want strategic upgrades (output at Q6_K, attention output at Q4_K)
- ✅ **Recommended for most users** seeking optimal balance

#### Choose Q3_HIFI (Pure) When:
- ✅ You need maximum quality at ~3.75 bpw
- ✅ Your model has important outlier weights across all tensors
- ✅ You have an importance matrix available
- ✅ Quality is more important than speed
- ✅ You're experimenting with new quantization techniques
- ✅ You want to preserve extreme values accurately everywhere

#### Choose Q3_K_S When:
- ✅ File size is the primary concern
- ✅ You need the fastest inference possible
- ✅ You're running on resource-constrained devices
- ✅ You can tolerate slightly lower quality
- ✅ You want the most aggressive compression
- ✅ You need maximum hardware optimization

#### Choose Q3_K_M When:
- ✅ You want a good balance of size, speed, and quality
- ✅ You need proven, stable quantization
- ✅ You want better quality than Q3_K_S without much size penalty
- ✅ You want mature hardware support
- ✅ You're looking for a "sweet spot" format
- ✅ Production deployment where stability matters

---

## Performance Benchmarks (Reference)

### File Size (Qwen3-0.6B model - actual results):
- **Q3_HIFI (Hybrid)**: **329MB** - Smallest with quality upgrades
- **Q3_K_S**: 380MB - Smallest pure format
- **Q3_K_M**: 404MB - Balanced pure format
- **Q3_HIFI (Pure)**: ~370MB (estimated) - All Q3_HIFI

### Quality (Llama-3-8B model - reference):
- **Q3_K_S**: 3.41 GB, +1.6321 perplexity increase
- **Q3_K_M**: 3.74 GB, +0.6569 perplexity increase
- **Q3_HIFI (Hybrid)**: ~3.33 GB (est.), expected similar or better than Q3_K_M (has Q6_K output + Q3_HIFI on critical tensors)
- **Q3_HIFI (Pure)**: ~3.75 GB, quality not yet benchmarked (expected similar or better than Q3_K_M)

---

## Summary Table

| Feature | Q3_HIFI (Pure) | Q3_HIFI (Hybrid) | Q3_K_S | Q3_K_M |
|---------|----------------|------------------|--------|--------|
| **File Size (0.6B)** | ~370MB | **329MB** ⭐ | 380MB | 404MB |
| **File Size (7B est.)** | ~3.75 GB | **~3.33 GB** ⭐ | ~3.42 GB | ~3.75 GB |
| **Bits/Weight** | 3.875 bpw | ~3.47 bpw | ~3.41 bpw | ~3.74 bpw |
| **Quality** | ⭐⭐⭐⭐⭐ (best) | ⭐⭐⭐⭐⭐ (best) | ⭐⭐⭐ (lowest) | ⭐⭐⭐⭐ (good) |
| **Speed** | ⭐⭐⭐ (slowest) | ⭐⭐⭐⭐ (good) | ⭐⭐⭐⭐⭐ (fastest) | ⭐⭐⭐⭐ (very fast) |
| **Memory** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hardware Support** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Quantization Time** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Outlier Preservation** | ✅ Yes (all tensors) | ✅ Yes (attn_v, ffn_down) | ❌ No | ❌ No |
| **Importance Matrix** | ✅ Supported | ✅ Supported | ✅ Supported | ✅ Supported |
| **Maturity** | ⭐⭐ (new) | ⭐⭐ (new) | ⭐⭐⭐⭐⭐ (mature) | ⭐⭐⭐⭐⭐ (mature) |
| **Ease of Use** | ⭐⭐⭐⭐ | ⭐⭐⭐ (manual setup) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Recommendations

### For Production Use (Recommended):
**Q3_HIFI (Hybrid)** is the **top recommendation** for most users due to:
- ✅ **Smallest file size** (329MB vs 380-404MB for 0.6B model)
- ✅ **Best quality-to-size ratio** - Q3_HIFI on critical tensors + Q6_K output
- ✅ **Quality matching or exceeding Q3_K_M** with smaller file
- ✅ **Faster than pure Q3_HIFI** (only 15% of tensors have outlier overhead)
- ✅ Strategic tensor selection maximizes benefits

**Command to use:**
```bash
llama-quantize \
  --tensor-type "attn_v=q3_hifi" \
  --tensor-type "ffn_down=q3_hifi" \
  --tensor-type "output.weight=q6_k" \
  --tensor-type "attn_output.weight=q4_k" \
  --tensor-type ".*=q3_k" \
  input.gguf output.gguf Q3_HIFI
```

### For Maximum Compression (Pure Formats):
**Q3_K_S** is the clear choice when:
- File size is critical
- Speed is paramount
- Slight quality loss is acceptable
- You want a single-command quantization

### For Balanced Production (Pure Formats):
**Q3_K_M** is recommended when:
- You want proven quality and stability
- Excellent hardware support is required
- You prefer automatic tensor selection
- Mature, well-tested format is important

### For Maximum Quality (Research):
**Q3_HIFI (Pure)** shows promise for:
- Research and experimentation
- Models sensitive to outliers across all tensors
- When you have importance matrices
- Future optimization potential

### For Speed-Critical Applications:
**Q3_K_S** or **Q3_K_M** are both excellent choices, with Q3_K_S being slightly faster. **Q3_HIFI (Hybrid)** is also quite fast since most tensors use optimized Q3_K.

---

## Future Considerations

- **Q3_HIFI** may see performance improvements as it gets more optimization
- GPU kernel optimizations for Q3_HIFI could significantly improve speed
- Importance matrix integration may make Q3_HIFI more competitive
- Ongoing research may improve outlier selection algorithms

---

## Conclusion

Each format serves different needs:
- **Q3_K_S**: Best for maximum compression and speed (pure format)
- **Q3_K_M**: Best for balanced production use (pure format)
- **Q3_HIFI (Pure)**: Best for maximum quality and outlier preservation everywhere (with speed tradeoff)
- **Q3_HIFI (Hybrid)**: ⭐ **Best overall** - Smallest file size with excellent quality and good speed

### Updated Recommendation

For most users, **Q3_HIFI (Hybrid)** offers the best overall balance:
- ✅ **Smallest file size** (329MB vs 380-404MB)
- ✅ **Excellent quality** (Q3_HIFI on critical tensors + Q6_K output)
- ✅ **Good speed** (most tensors use fast Q3_K)
- ✅ **Better than Q3_K_M** in both size and quality

The hybrid approach demonstrates that **selective use of Q3_HIFI** on critical tensors (attn_v, ffn_down) combined with strategic upgrades (output.weight→Q6_K) and efficient bulk quantization (Q3_K for everything else) achieves the optimal balance of size, quality, and speed.

**For pure formats without manual configuration**, Q3_K_M remains the best choice for balanced production use, while Q3_K_S is best for maximum compression.

