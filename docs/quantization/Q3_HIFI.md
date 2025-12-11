# Q3_HIFI Quantization Format

## Overview

**Q3_HIFI** is a 3-bit quantization format that combines the speed of Q3_K with improved quality through selective FP16 outlier preservation. It achieves **~98% of Q3_K_M speed** while delivering **17% better perplexity** and **smaller file size**.

## Key Features

| Feature | Value |
|---------|-------|
| Bits per weight | ~4.0 bpw |
| Block size | 256 weights |
| Outliers per block | 6 (FP16) |
| Block structure | Q3_K-compatible + outlier tail |

## Performance Comparison

Tested on Qwen3-1.7B:

| Format | Size | Perplexity | Speed | vs Q3_K_M |
|--------|------|------------|-------|-----------|
| Q3_K_S | 949 MiB | 21.61 | 24.2 tok/s | baseline |
| Q3_K_M | 1018 MiB | 20.25 | 24.7 tok/s | baseline |
| **Q3_HIFI** | **991 MiB** | **16.66** | **24.6 tok/s** | ✅ Better quality, smaller |

## Block Structure

```c
typedef struct {
    // === Q3_K-COMPATIBLE REGION (110 bytes) ===
    uint8_t hmask[32];     // 32 bytes: high bit mask (1 bit per weight)
    uint8_t qs[64];        // 64 bytes: low 2 bits (2 bits per weight)
    uint8_t scales[12];    // 12 bytes: 16 sub-group scales (6-bit each)
    ggml_half d;           // 2 bytes: super-block scale
    
    // === OUTLIER EXTENSION (18 bytes) ===
    uint8_t outlier_idx[6];    // 6 bytes: outlier positions (0-255)
    ggml_half outlier_vals[6]; // 12 bytes: FP16 outlier values
} block_q3_hifi;  // Total: 128 bytes
```

## How It Works

### Quantization
1. Identify the 6 weights with highest magnitude × importance (from imatrix)
2. Store these outliers as exact FP16 values
3. Set outlier positions to zero in the Q3_K bulk data
4. Quantize remaining weights using standard Q3_K encoding

### Inference (vec_dot)
1. Compute Q3_K-style bulk dot product (pre-zeroed outliers contribute 0)
2. Add outlier corrections: `sum += outlier_val[k] * activation[outlier_idx[k]]`

### Why Pre-Zeroing Works
By storing zero at outlier positions during quantization, the bulk SIMD dot product naturally skips outliers. This eliminates the need for subtraction during inference.

## Usage

### Creating a Q3_HIFI Model

```bash
# Basic quantization
./llama-quantize model-f16.gguf model-q3hifi.gguf Q3_HIFI

# With importance matrix (recommended)
./llama-quantize --imatrix imatrix.gguf model-f16.gguf model-q3hifi.gguf Q3_HIFI
```

### Running Inference

```bash
# CPU inference
./llama-cli -m model-q3hifi.gguf -p "Hello" -n 100

# GPU inference (CUDA)
./llama-cli -m model-q3hifi.gguf -p "Hello" -n 100 -ngl 99

# GPU inference (Metal)
./llama-cli -m model-q3hifi.gguf -p "Hello" -n 100 -ngl 99
```

### Benchmarking

```bash
# Speed benchmark
./llama-bench -m model-q3hifi.gguf -t 4 -r 3 -p 0 -n 20

# Perplexity evaluation
./llama-perplexity -m model-q3hifi.gguf -f wikitext-2-raw/wiki.test.raw
```

## Backend Support

| Backend | Dequantization | vec_dot | Status |
|---------|----------------|---------|--------|
| CPU (AVX2) | ✅ | ✅ | Full support |
| CPU (NEON) | ✅ | ✅ | Full support |
| CUDA | ✅ | ✅ | Full support |
| Metal | ✅ | ✅ | Full support |
| SYCL | ✅ | ✅ | Full support |
| Vulkan | ✅ | ✅ | Full support |

## When to Use Q3_HIFI

### ✅ Recommended For:
- Memory-constrained deployments where Q4 is too large
- Quality-critical 3-bit quantization needs
- Edge devices with limited RAM but decent compute

### ❌ Consider Alternatives If:
- Maximum speed is critical → use Q3_K_M
- Quality is paramount → use Q4_K_M or higher
- Very large models (70B+) → test perplexity carefully

## Technical Details

### Outlier Selection Algorithm
1. Compute importance score: `score[i] = |weight[i]| × imatrix[i]`
2. Select top-6 positions by score
3. Store exact FP16 values at those positions

### Memory Layout Compatibility
The first 110 bytes of `block_q3_hifi` exactly match `block_q3_K`, enabling:
- Reuse of optimized Q3_K SIMD kernels
- Minimal code changes for backend support
- Zero-copy bulk dot product computation

### Performance Optimizations
1. **Loop unrolling**: 6 outliers unrolled in vec_dot
2. **Pre-zeroing**: Outliers set to 0 during quantization
3. **SIMD-friendly layout**: Q3_K-compatible bit packing

## References

- [llama.cpp Quantization Guide](../build.md)
- [Q3_K Implementation](../../ggml/src/ggml-quants.c)
- [Original GPTQ Paper](https://arxiv.org/abs/2210.17323)

