# Q3 Quantization Formats Comparison: Q3_HIFI vs Q3_K_S vs Q3_K_M

## Executive Summary

This document compares three 3-bit quantization strategies available in llama.cpp:
- **Q3_HIFI**: A hybrid format using 3-bit quantization with FP16 outliers
- **Q3_K_S**: Aggressive mixed quantization using Q3_K format for most tensors
- **Q3_K_M**: Balanced mixed quantization using Q3_K format with more conservative tensor selection

---

## Technical Specifications

### Q3_HIFI
- **Format**: Hybrid 3-bit + FP16 outliers
- **Block Structure**: 256 weights per block
  - 250 weights: 3-bit quantized (96 bytes)
  - 6 weights: Stored as FP16 outliers (12 bytes)
  - 6 outlier indices: uint16_t (12 bytes)
  - 1 float scale: 4 bytes
- **Bits per Weight**: ~3.875 bpw (124 bytes / 256 weights × 8)
- **Block Size**: 124 bytes per 256 weights
- **Outlier Strategy**: Identifies top-6 outliers by magnitude (optionally weighted by importance matrix) and stores them in full FP16 precision

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

| Format | Bits per Weight | File Size (7B model) | Notes |
|--------|----------------|---------------------|-------|
| **Q3_HIFI** | 3.875 bpw | ~3.75 GB | Slightly larger due to outlier storage |
| **Q3_K_S** | ~3.41 bpw (mixed) | ~3.42 GB | Smallest, most aggressive |
| **Q3_K_M** | ~3.74 bpw (mixed) | ~3.75 GB | Similar to Q3_HIFI in size |

**Winner**: Q3_K_S (smallest), Q3_K_M and Q3_HIFI are similar

### 2. Quality / Accuracy

#### Q3_HIFI
- **Pros**:
  - Preserves critical outliers in full FP16 precision
  - Can use importance matrix to intelligently select outliers
  - Better preservation of extreme values that might be important
  - Potentially better for models with sparse important weights
  
- **Cons**:
  - Fixed 6 outliers per block (may not be optimal for all distributions)
  - Outlier selection is magnitude-based (though can be weighted)
  - Slightly more complex dequantization

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

**Winner**: Q3_HIFI (potentially best for outlier-sensitive models), Q3_K_M (best proven quality)

### 3. Speed / Performance

#### Q3_HIFI
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

**Winner**: Q3_K_S (fastest), Q3_K_M (very close), Q3_HIFI (slowest due to outlier handling)

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

#### Q3_HIFI
- **RAM**: Slightly higher due to outlier storage
- **VRAM**: Similar to Q3_K_M
- **Cache**: Less efficient (scattered outlier access)

#### Q3_K_S
- **RAM**: Lowest
- **VRAM**: Lowest
- **Cache**: Most efficient

#### Q3_K_M
- **RAM**: Similar to Q3_HIFI
- **VRAM**: Similar to Q3_HIFI
- **Cache**: Good (better than Q3_HIFI)

**Winner**: Q3_K_S (lowest memory)

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

#### Choose Q3_HIFI When:
- ✅ You need maximum quality at ~3.75 bpw
- ✅ Your model has important outlier weights
- ✅ You have an importance matrix available
- ✅ Quality is more important than speed
- ✅ You're experimenting with new quantization techniques
- ✅ You want to preserve extreme values accurately

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

Based on Llama-3-8B model:
- **Q3_K_S**: 3.41 GB, +1.6321 perplexity increase
- **Q3_K_M**: 3.74 GB, +0.6569 perplexity increase
- **Q3_HIFI**: ~3.75 GB, quality not yet benchmarked (expected similar or better than Q3_K_M)

---

## Summary Table

| Feature | Q3_HIFI | Q3_K_S | Q3_K_M |
|---------|---------|--------|--------|
| **File Size** | ~3.75 GB | ~3.42 GB | ~3.75 GB |
| **Bits/Weight** | 3.875 bpw | ~3.41 bpw | ~3.74 bpw |
| **Quality** | ⭐⭐⭐⭐⭐ (best) | ⭐⭐⭐ (lowest) | ⭐⭐⭐⭐ (good) |
| **Speed** | ⭐⭐⭐ (slowest) | ⭐⭐⭐⭐⭐ (fastest) | ⭐⭐⭐⭐ (very fast) |
| **Memory** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Hardware Support** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Quantization Time** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Outlier Preservation** | ✅ Yes (6 per block) | ❌ No | ❌ No |
| **Importance Matrix** | ✅ Supported | ✅ Supported | ✅ Supported |
| **Maturity** | ⭐⭐ (new) | ⭐⭐⭐⭐⭐ (mature) | ⭐⭐⭐⭐⭐ (mature) |

---

## Recommendations

### For Production Use:
**Q3_K_M** is recommended for most production scenarios due to:
- Proven quality and stability
- Excellent hardware support
- Good balance of all factors
- Mature, well-tested format

### For Maximum Compression:
**Q3_K_S** is the clear choice when:
- File size is critical
- Speed is paramount
- Slight quality loss is acceptable

### For Maximum Quality:
**Q3_HIFI** shows promise for:
- Research and experimentation
- Models sensitive to outliers
- When you have importance matrices
- Future optimization potential

### For Speed-Critical Applications:
**Q3_K_S** or **Q3_K_M** are both excellent choices, with Q3_K_S being slightly faster.

---

## Future Considerations

- **Q3_HIFI** may see performance improvements as it gets more optimization
- GPU kernel optimizations for Q3_HIFI could significantly improve speed
- Importance matrix integration may make Q3_HIFI more competitive
- Ongoing research may improve outlier selection algorithms

---

## Conclusion

Each format serves different needs:
- **Q3_K_S**: Best for maximum compression and speed
- **Q3_K_M**: Best for balanced production use
- **Q3_HIFI**: Best for maximum quality and outlier preservation (with speed tradeoff)

The choice depends on your priorities: size, speed, or quality. For most users, **Q3_K_M** offers the best overall balance, while **Q3_HIFI** is worth exploring if quality is paramount and you can accept the speed tradeoff.

