# Q3_HIFI Optimization Plan v2

**Mission:** Create a quantization format that is **smaller**, **faster**, AND **higher quality** than Q3_K_M.

**Critical Rule:** Every change must be validated. Changes that cause regression in size, speed, OR quality must be reverted or fixed before proceeding.

---

## Executive Summary

### Target Metrics (vs Q3_K_M baseline)
| Metric | Q3_K_M | Target | Constraint |
|--------|--------|--------|------------|
| File Size | ~1018 MiB | ≤ 1018 MiB | **Must not be larger** |
| Perplexity | ~22.78 | < 22.78 | **Must be better** |
| Speed | ~100 tok/s | > 50 tok/s | **Within 2x** |

### Block Budget Analysis

**Q3_K block (110 bytes per 256 weights = 3.44 BPW):**
- hmask: 32 bytes (1 bit per weight for sign)
- qs: 64 bytes (2 bits per weight)
- scales: 12 bytes (per-16 subscales)
- d: 2 bytes (FP16 scale)

**Q3_HIFI v7 block (current: 116 bytes = 3.625 BPW):** ✅ ACHIEVED
- d: 2 bytes ✅ (FP16 scale)
- ql: 64 bytes ✅ (2 bits per weight, SIMD-friendly)
- qh: 32 bytes ✅ (1 bit per weight, SIMD-friendly)
- outlier_idx: 6 bytes ✅ (uint8)
- outlier_vals: 12 bytes (FP16)

**Q3_HIFI v5 target (107 bytes = 3.34 BPW):** 🎯 NEXT
- d: 2 bytes (FP16 scale)
- qs: 96 bytes (3 bits per weight)
- outlier_idx: 6 bytes (uint8)
- outlier_codes: 3 bytes (4-bit codebook indices) - saves 9 bytes!

---

## Phase 0: Baseline Verification

### Step 0.1: Document Current State
**Goal:** Establish exact baseline numbers for ALL metrics

**Tasks:**
- [ ] Measure current Q3_HIFI file size
- [ ] Measure current Q3_HIFI perplexity (full test, not just 20 chunks)
- [ ] Measure current Q3_HIFI speed
- [ ] Document exact block structure and size

**Commands:**
```powershell
# Build
cmake --build build --config Release

# Create fresh quantized model
.\build\bin\Release\llama-quantize.exe --imatrix .\qwen3-1.7b-imatrix.gguf `
    .\Qwen3-1.7B-f16.gguf .\Qwen3-1.7B-Q3_HIFI-baseline.gguf Q3_HIFI

# Measure file size
(Get-Item .\Qwen3-1.7B-Q3_HIFI-baseline.gguf).Length / 1MB

# Measure perplexity (full test for accuracy)
.\build\bin\Release\llama-perplexity.exe -m .\Qwen3-1.7B-Q3_HIFI-baseline.gguf `
    -f .\wikitext-2-raw\wikitext-2-raw\wiki.test.raw --ppl-stride 0 -c 512

# Measure speed (short run for speed)
.\build\bin\Release\llama-cli.exe -m .\Qwen3-1.7B-Q3_HIFI-baseline.gguf `
    -p "Hello" -n 100 2>&1 | Select-String "tok/s"
```

**Baseline Results (Updated 2025-12-11):**
| Metric | Q3_K_M | Q3_HIFI v7 | Notes |
|--------|--------|------------|-------|
| File Size | 1023.52 MiB | **987.37 MiB** | ✅ 36 MiB smaller! |
| Block Size | 110 bytes | 116 bytes | +6 bytes (was 124) |
| Block Layout | ql[64]+qh[32]+scales | ql[64]+qh[32]+outliers | Split layout |
| BPW | 3.44 | 3.62 | |
| Perplexity | 22.78 | **21.91** | ✅ Better quality! |
| Speed | ~56 tok/s | 9 tok/s | ⚠️ 6x slower |
| Quant Time | - | 11s | ✅ 2x faster than v4 |

**Key Optimizations Applied:**
- ✅ FP16 scale (saved 2 bytes)
- ✅ uint8 outlier indices (saved 6 bytes)
- ✅ Split ql/qh layout (SIMD-friendly, 2x faster quant)
- ✅ AVX2 vec_dot (correct, but extraction still scalar)

---

## Phase 1: Size Optimization (Critical Path)

The current Q3_HIFI block is **8 bytes larger** than Q3_K. This MUST be fixed first.

### Step 1.1: Use FP16 Scale (Save 2 bytes)
**Goal:** Change `float d` to `ggml_fp16_t d`

**Current:** `float d` (4 bytes)
**Target:** `ggml_fp16_t d` (2 bytes)

**Risk:** Minimal - FP16 has sufficient precision for scale factors

**Files to modify:**
- `ggml/include/ggml.h` - block_q3_hifi structure
- `ggml/src/ggml-quants.c` - quantize/dequantize functions
- `ggml/src/ggml-cpu/quants.c` - vec_dot functions
- `ggml/src/ggml-cpu/arch/x86/quants.c` - AVX2 implementations
- GPU shaders (Vulkan, CUDA, Metal)

**Verification:**
- [ ] Block size: 118 → 116 bytes
- [ ] Perplexity: Should be unchanged (< 0.1 difference)
- [ ] Speed: Should be unchanged or slightly faster (fewer bytes to load)

**Go/No-Go Gate:**
- ✅ Proceed if: Perplexity unchanged, size reduced
- ❌ Revert if: Perplexity increases by > 0.1

---

### Step 1.2: Implicit Outlier Indices (Save 6 bytes) ⚡ REVOLUTIONARY
**Goal:** Eliminate explicit storage of outlier indices

**Concept:** Instead of storing 6 indices (6 bytes), encode outlier positions implicitly:
1. During quantization: Set the 3-bit value at outlier positions to a RESERVED value (e.g., all 1s = 7)
2. During dequantization: Any position with value 7 is an outlier → look up FP16 value
3. Store outlier FP16 values in sorted order (by position), so we know which maps to which

**Implementation:**
```c
// Quantization: Mark outlier positions with sentinel value
for (int i = 0; i < Q3_HIFI_BLOCK_SIZE; ++i) {
    if (is_outlier[i]) {
        set_q3_value(block, i, 7);  // Sentinel value = max (all bits set)
    } else {
        int q = quantize_to_3bit(x[i], scale);
        if (q == 7) q = 6;  // Clamp non-outliers to avoid collision
        set_q3_value(block, i, q);
    }
}

// Dequantization: Check for sentinel
int q3 = get_q3_value(block, i);
if (q3 == 7) {
    // This is an outlier - find its FP16 value
    y[i] = get_next_outlier_value(block, &outlier_counter);
} else {
    y[i] = (q3 - 4) * scale;  // Normal: maps [0,6] → [-4,2]
}
```

**Trade-offs:**
- ✅ Saves 6 bytes per block (5% size reduction)
- ✅ Reduces cache pressure during inference
- ⚠️ Reduces quantization levels from 8 to 7 for non-outliers
- ⚠️ Requires scanning for outliers during dequant (minor overhead)

**Risk Assessment:**
- Quality impact: Unknown - need to test if 7 levels vs 8 matters
- Speed impact: Likely minor slowdown during dequant (sentinel check)

**Verification:**
- [ ] Block size: 116 → 110 bytes (matches Q3_K!)
- [ ] Perplexity: Target < 0.5 degradation
- [ ] Speed: Target < 10% slowdown

**Go/No-Go Gate:**
- ✅ Proceed if: Perplexity degradation < 0.5, size savings achieved
- ❌ Revert if: Perplexity degradation > 0.5

---

### Step 1.3: Alternative - Packed Indices (Save 3 bytes)
**Goal:** If implicit indices hurt quality, try packed storage instead

**Concept:** Pack 6 indices (each 0-255) more efficiently:
- Current: 6 × 8 bits = 48 bits = 6 bytes
- Packed: 6 × 8 bits = 48 bits (no savings possible with uint8)
- Alternative: Use bitmap for common positions

**Alternative Idea - Position Bitmap:**
- Store a 256-bit bitmap (32 bytes) indicating outlier positions
- This is WORSE for 6 outliers (32 vs 6 bytes)

**Conclusion:** Stick with current uint8 indices OR use implicit approach (Step 1.2)

---

## Phase 2: Quality Verification

### Step 2.1: Establish Quality Baseline
**Goal:** Ensure quantization algorithm is correct

**Tests:**
1. Round-trip test: quantize → dequantize → compare MSE
2. Outlier preservation: outliers should be exact FP16
3. Dot product accuracy: vec_dot vs dequantized dot product

**Create test file: `tests/test-q3-hifi.cpp`**

```cpp
// Test 1: Round-trip MSE
void test_roundtrip_mse() {
    float input[256];
    fill_random(input);
    
    block_q3_hifi block;
    quantize_row_q3_hifi_ref(input, &block, 256);
    
    float output[256];
    dequantize_row_q3_hifi(&block, output, 256);
    
    float mse = compute_mse(input, output, 256);
    ASSERT(mse < 0.01);  // Reasonable MSE threshold
}

// Test 2: Outlier preservation
void test_outlier_preservation() {
    // Create input with known outliers
    float input[256] = {0};
    input[0] = 100.0f;   // Large outlier
    input[128] = -50.0f; // Negative outlier
    
    block_q3_hifi block;
    quantize_row_q3_hifi_ref(input, &block, 256);
    
    float output[256];
    dequantize_row_q3_hifi(&block, output, 256);
    
    // Outliers should be preserved exactly (FP16 precision)
    ASSERT(abs(output[0] - input[0]) < 0.01);
    ASSERT(abs(output[128] - input[128]) < 0.01);
}

// Test 3: Dot product accuracy
void test_dot_product() {
    float x[256], y[256];
    fill_random(x);
    fill_random(y);
    
    block_q3_hifi x_q;
    block_q8_K y_q;
    quantize_row_q3_hifi_ref(x, &x_q, 256);
    quantize_row_q8_K_ref(y, &y_q, 256);
    
    float result;
    ggml_vec_dot_q3_hifi_q8_K(256, &result, 0, &x_q, 0, &y_q, 0, 1);
    
    // Dequantize and compute reference
    float x_deq[256], y_deq[256];
    dequantize_row_q3_hifi(&x_q, x_deq, 256);
    dequantize_row_q8_K(&y_q, y_deq, 256);
    float ref = dot_product(x_deq, y_deq, 256);
    
    float rel_error = abs(result - ref) / abs(ref);
    ASSERT(rel_error < 0.001);  // 0.1% tolerance
}
```

---

### Step 2.2: Review Outlier Selection
**Goal:** Ensure outliers are chosen optimally

**Current algorithm:**
```c
// Find top-6 by magnitude
for (k = 0; k < 6; k++) {
    argmax over all positions
    mark as outlier
}
```

**Potential improvements:**
1. **iMatrix weighting:** `score[i] = |x[i]| * imatrix[i]`
2. **MSE-based selection:** Choose outliers that maximize MSE reduction
3. **Gradient-aware:** If available, use sensitivity information

**Verification:**
- Compare perplexity with different selection strategies
- Document best approach

---

## Phase 3: Speed Optimization

### Step 3.1: Profile Current Implementation
**Goal:** Identify actual bottlenecks

**Use Windows Performance Analyzer or Visual Studio Profiler:**
```powershell
# Profile with VS tools
.\build\bin\Release\llama-perplexity.exe -m .\Qwen3-1.7B-Q3_HIFI-baseline.gguf `
    -f .\wikitext-2-raw\wikitext-2-raw\wiki.test.raw -c 512 --chunks 10
```

**Expected hotspots:**
1. 3-bit extraction (bit manipulation)
2. Outlier correction loop
3. Memory loads

---

### Step 3.2: Format Change to Split ql/qh Layout ⚡ CRITICAL FOR SPEED
**Goal:** Enable efficient SIMD bit extraction like Q3_K

**Current Problem:**
Our `qs[96]` continuous 3-bit packing is **fundamentally SIMD-unfriendly**:
```c
// Current: bits cross byte boundaries - requires complex extraction
const int byte_idx = (i * 3) / 8;
const int bit_offset = (i * 3) % 8;
uint8_t bits = (qs[byte_idx] >> bit_offset) & 7;
if (bit_offset > 5) bits |= (qs[byte_idx + 1] << (8 - bit_offset)) & 7;
```

**Q3_K's Approach (split layout):**
```c
// Q3_K: simple masks, SIMD-friendly
int low = (ql[i/4] >> ((i%4)*2)) & 0x03;   // 2 bits from ql[64]
int high = (qh[i/8] >> (i%8)) & 0x01;       // 1 bit from qh[32]
int value = (low | (high << 2)) - 4;
```

**Why Split Layout is ~5x Faster:**
| Operation | Continuous 3-bit | Split ql/qh |
|-----------|------------------|-------------|
| Byte alignment | Crosses boundaries | Always aligned |
| SIMD extraction | Requires scalar loop | Pure vector ops |
| Bits per vector | Complex packing | Simple masks |

**Proposed New Block Structure (116 bytes, same size):**
```c
typedef struct {
    ggml_fp16_t d;                    // 2 bytes
    uint8_t  ql[64];                  // 64 bytes (2 bits per weight)
    uint8_t  qh[32];                  // 32 bytes (1 bit per weight)
    uint8_t  outlier_idx[6];          // 6 bytes
    ggml_fp16_t outlier_vals[6];      // 12 bytes
} block_q3_hifi_v2;                   // Total: 116 bytes (same as current!)
```

**Expected Speed Improvement:**
| Metric | Current (qs[96]) | After (ql/qh) |
|--------|------------------|---------------|
| Speed | 10 tok/s | **40-50 tok/s** |
| vs Q3_K_M | 5.6x slower | **1.1-1.4x slower** |

**Implementation Steps:**
1. Change block structure to split layout
2. Update quantize/dequantize functions
3. Rewrite AVX2 vec_dot with simple bit extraction
4. Re-quantize all models

**Risk:** Breaking change - all existing Q3_HIFI models need re-quantization

---

### Step 3.3: Pre-Zero Outliers During Quantization ⚡ KEY OPTIMIZATION
**Goal:** Eliminate runtime outlier handling in vec_dot

**Current Problem:**
```c
// Current vec_dot: compute full sum, then correct for outliers
int32_t sum_bulk = simd_dot_product(q3, q8);
for (int k = 0; k < 6; ++k) {
    sum_bulk -= q3[outlier_idx[k]] * q8[outlier_idx[k]];  // SUBTRACT
    outlier_correction += outlier_val[k] * q8[outlier_idx[k]];  // ADD
}
```
This requires **subtracting the bulk contribution at outlier positions** - extra work!

**Solution: Store 0 at outlier positions during quantization**
```c
// During quantization:
for (int i = 0; i < 256; ++i) {
    if (is_outlier[i]) {
        set_q3_value(block, i, 4);  // Store 4 → maps to 0 after -4 bias
    } else {
        set_q3_value(block, i, quantize(x[i]));
    }
}
```

**Optimized vec_dot (no subtraction needed!):**
```c
int32_t sum_bulk = simd_dot_product(q3, q8);  // Outliers contribute 0!
// Just add outlier corrections:
for (int k = 0; k < 6; ++k) {
    outlier_correction += outlier_val[k] * q8[outlier_idx[k]];
}
```

**Benefits:**
- Eliminates 6 subtract operations per block
- Cleaner SIMD code path
- No need to track outlier positions during dot product

**Status:** ⚠️ Requires quantization code change - low priority until format change (3.2) is done

---

### Step 3.4: Fused MatMul Kernel ⚡ REVOLUTIONARY
**Goal:** Compute directly on quantized data without dequantize step

**Current flow:**
```
Q3_HIFI block → dequantize to float[256] → multiply with Q8 → accumulate
```

**Fused flow:**
```
Q3_HIFI block + Q8 block → direct integer multiply → scale at end
```

**Implementation for vec_dot:**
```c
// Process entire block without dequantization buffer
int32_t sum = 0;
for (int i = 0; i < 256; i += 32) {
    // Extract 32 q3 values
    int8_t q3[32];
    extract_q3_values(block->ql, block->qh, i, q3);
    
    // Load 32 q8 values
    const int8_t* q8 = y[ib].qs + i;
    
    // Integer dot product
    sum += dot_product_int8(q3, q8, 32);
}

// Apply scales
float result = sum * block->d * y[ib].d;

// Add outlier corrections (these need special handling)
for (int k = 0; k < 6; k++) {
    int idx = block->outlier_idx[k];
    float outlier_val = fp16_to_f32(block->outlier_vals[k]);
    float q3_val = get_q3_value(block, idx) * block->d;
    result += (outlier_val - q3_val) * (y[ib].qs[idx] * y[ib].d);
}
```

**Verification:**
- Unit test MUST pass before perplexity test
- Any difference indicates a bug

---

## Phase 4: Revolutionary Ideas (High Risk/Reward)

### Step 4.1: Reduce Block Size to 128 ⚡ EXPERIMENTAL
**Goal:** Better cache locality, faster processing

**Current:** 256 values per block, 6 outliers
**Proposed:** 128 values per block, 3 outliers

**Block size comparison:**
| Layout | 256-block | 128-block | Notes |
|--------|-----------|-----------|-------|
| d (FP16) | 2 bytes | 2 bytes | |
| ql | 64 bytes | 32 bytes | |
| qh | 32 bytes | 16 bytes | |
| outlier_idx | 6 bytes | 3 bytes | |
| outlier_vals | 12 bytes | 6 bytes | |
| **Total** | 116 bytes | 59 bytes | |
| **BPW** | 3.625 | 3.6875 | Slight increase |

**Trade-off:** More overhead per value, but:
- Better L1 cache utilization
- Smaller SIMD working set
- Potentially faster outlier lookup

**Risk:** Q8_K uses 256-block size. Would need Q8_128 or padding.

**Decision:** DEFER until other optimizations complete

---

### Step 4.2: Hybrid Outlier Format ⚡ EXPERIMENTAL
**Goal:** Reduce outlier storage while maintaining quality

**Current:** 6 × FP16 values = 12 bytes
**Proposed:** 6 × (sign + 8-bit magnitude) = 6 bytes

**Implementation:**
```c
// Quantization
for each outlier i:
    float val = x[outlier_idx[i]];
    int8_t sign = (val < 0) ? -1 : 1;
    float magnitude = fabsf(val);
    uint8_t rank = quantize_log_scale(magnitude, block_max);
    outlier_packed[i] = (sign < 0 ? 0x80 : 0) | rank;

// Dequantization
float val = dequantize_log_scale(outlier_packed[i] & 0x7F, block_max);
if (outlier_packed[i] & 0x80) val = -val;
```

**Risk:** HIGH - Log-scale quantization of outliers may hurt quality significantly

**Verification Required:**
- Test on multiple models
- Compare perplexity carefully
- Only proceed if degradation < 0.3 PPL

---

### Step 4.3: Static Outlier Positions (from iMatrix) ⚡ EXPERIMENTAL
**Goal:** Determine outlier positions at quantization time based on importance

**Concept:**
1. Use iMatrix to identify globally important weight positions
2. Store fixed outlier positions per tensor (not per block)
3. Reduces per-block overhead significantly

**Implementation:**
```c
// During quantization (once per tensor):
int static_outlier_positions[6]; // Fixed for entire tensor
find_most_important_positions(imatrix, static_outlier_positions);

// Per-block: only store the FP16 values
block->outlier_vals[6]; // 12 bytes, no indices needed
```

**Benefits:**
- Eliminates 6 bytes per block for indices
- Outlier positions are more "globally optimal"

**Risks:**
- Different blocks may have different outlier patterns
- May reduce effectiveness of outlier preservation

---

## Phase 4B: New Revolutionary Ideas (Added 2025-12-11) 🔥

### Summary of New Ideas

| Idea | Speed Gain | Size Gain | Accuracy Risk | Feasibility | Priority |
|------|-----------|----------|----------------|-------------|----------|
| **Learned Outlier Codes** | +15% | **-75% outlier storage** | Low | ✅ High | **#1** |
| **Predictive Outlier Skipping** | **+10-20%** | +1 byte | Very Low | ✅ High | **#2** |
| **Fuse into Q8_K** | **+50-100%** | **-100% outliers** | Low (with imatrix) | ⚠️ Medium | **#3** |

---

### 🔥 Step 4B.1: Learned Outlier Codes ⚡ PRIORITY 1 (Low Risk, High Reward)
**Goal:** Replace FP16 outliers with 4-bit codebook indices

**Current:** 6 × FP16 values = 12 bytes  
**Proposed:** 6 × 4-bit codes = 3 bytes + shared global codebook

**Concept:**
Instead of storing raw FP16 outlier values, cluster all outliers across the model 
into 16 prototype values and store 4-bit indices into this codebook.

**Implementation:**
```c
// Global codebook (shared across all blocks, learned from imatrix data)
static const float OUTLIER_CODEBOOK[16] = {
    -8.0f, -4.0f, -2.0f, -1.0f, -0.5f, -0.25f, -0.125f, 0.0f,
    0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f
};

// New block structure (107 bytes - smaller than Q3_K!)
typedef struct {
    ggml_fp16_t d;                    // 2 bytes
    uint8_t  qs[96];                  // 96 bytes (3-bit packed)
    uint8_t  outlier_idx[6];          // 6 bytes
    uint8_t  outlier_codes[3];        // 3 bytes (6 × 4-bit packed)
} block_q3_hifi_v3;

// Quantization: assign each outlier to nearest code
for (int k = 0; k < 6; k++) {
    float normalized = outlier_val[k] / block_scale;
    int code = find_nearest_codebook_entry(normalized, OUTLIER_CODEBOOK);
    pack_4bit(outlier_codes, k, code);
}

// Dequantization: simple table lookup
float outlier = OUTLIER_CODEBOOK[get_4bit(outlier_codes, k)] * block_scale;
```

**Expected Gains:**
- Outlier storage: 12 → 3 bytes (75% reduction)
- Block size: 116 → 107 bytes (smaller than Q3_K at 110!)
- BPW: 4.08 → ~3.9
- Faster: No FP16 conversion, just table lookup

**Risk:** LOW - 16 levels sufficient for outliers
**Validation:** Build optimal codebook from imatrix-weighted outlier histogram

---

### 🔥 Step 4B.2: Predictive Outlier Skipping ⚡ PRIORITY 2 (Medium Risk, Speed Gain)
**Goal:** Skip outlier correction dynamically at runtime

**Problem:** Always restoring 6 outliers/block, even when not strongly activated.

**Concept:**
Add a lightweight activation hint per block that predicts whether outlier 
correction is needed for typical inputs.

**Implementation:**
```c
// Add 1 byte to block
typedef struct {
    ggml_fp16_t d;
    uint8_t  qs[96];
    uint8_t  outlier_idx[6];
    ggml_fp16_t outlier_vals[6];
    uint8_t  activation_hint;  // 2-bit class: 0=skip, 1-3=apply with weight
} block_q3_hifi_adaptive;

// During quantization, compute expected outlier contribution:
float expected_contrib = 0;
for (int k = 0; k < 6; k++) {
    expected_contrib += fabsf(outlier_val[k]) * avg_activation * imatrix_weight[idx];
}
block->activation_hint = (expected_contrib > threshold) ? 1 : 0;

// In vec_dot (branch predictor-friendly):
if (block->activation_hint) {
    // Apply outlier correction only when predicted necessary
    apply_outlier_corrections(sum, block, q8);
}
```

**Expected Gains:**
- 10-20% speedup on average inputs
- Near-zero accuracy loss

**Note:** This is **input-adaptive quantization** - revolutionary!

---

### 🔥 Step 4B.3: Fuse Outliers into Q8_K ⚡ PRIORITY 3 (High Complexity, Maximum Gain)
**Goal:** Eliminate outlier overhead entirely via tensor co-design

**Problem:** vec_dot loads both Q3_HIFI and Q8_K, causing cache thrashing.

**Concept:**
When quantizing activations (Q8_K), embed outlier corrections directly:
1. Zero out Q8 positions corresponding to Q3_HIFI outliers
2. Pre-compute outlier products and add to bias term
3. vec_dot becomes pure bulk operation

**Implementation:**
```c
// During Q8_K quantization (given known Q3_HIFI outlier positions):
float correction = 0;
for (int k = 0; k < 6; k++) {
    int idx = weight_block->outlier_idx[k];
    correction += weight_block->outlier_val[k] * activation[idx];
    q8_block->qs[idx] = 0;  // Mask out in Q8
}
q8_block->correction = correction;  // Store per-block

// Now vec_dot is pure SIMD:
float sum = vec_dot_pure_bulk(q3_hifi, q8_k);  // No outlier loop!
sum += q8_block->correction;  // Single addition
```

**Expected Gains:**
- Eliminates 100% of outlier runtime overhead
- Enables pure SIMD vec_dot
- Model becomes smaller (no outlier vals in weights)

**Risks:**
- Only for matmul with bias (most operations qualify)
- Requires joint weight+activation quantization
- Needs imatrix (which we have)

**Note:** Co-designed scheme like SpQR but simpler!

---

## Revised Priority Order (Updated 2025-12-11)

Based on analysis of actual bottlenecks:

### Tier 1: Completed ✅
| Step | Description | Size Impact | Speed Impact | Status |
|------|-------------|-------------|--------------|--------|
| ✅ 1.1 | FP16 scale | -2 bytes | None | Done |
| ✅ 1.1b | uint8 outlier_idx | -6 bytes | None | Done |
| ✅ 3.1 | AVX2 vec_dot (basic) | None | +38% (7→10 tok/s) | Done |
| ✅ 3.2 | Split ql/qh format | None | +2x quant speed | Done |

### Tier 2: Next Steps (Speed)
| Step | Description | Size Impact | Speed Impact |
|------|-------------|-------------|--------------|
| 3.4 | Pure SIMD extraction | None | +5x (target 50 tok/s) |
| 3.3 | Pre-zero outliers | None | +10-20% |

### Tier 3: Size Optimization  
| Step | Description | Size Impact | Speed Impact |
|------|-------------|-------------|--------------|
| **4B.1** | **Learned Outlier Codes** | **-9 bytes** | +5% |

### Tier 4: Research (High Complexity)
| Step | Description | Size Impact | Speed Impact |
|------|-------------|-------------|--------------|
| 4B.3 | Fuse into Q8_K | -12 bytes | +50%+ |
| 4B.2 | Predictive Skipping | +1 byte | +10-20% |

### Key Insight (Updated):
**Step 3.2 (split ql/qh format) is complete but didn't provide speed gains** because extraction is still scalar. For Q3_K-level speed, we need:
- **Pure SIMD extraction** using shuffle/blend operations (complex)
- **Or: Accept 6x slower speed** in exchange for better quality (PPL 21.9 vs 22.8)

---

## Phase 5: Testing Protocol

### For Each Change:

1. **Before implementing:**
   - Document expected impact on size, speed, quality
   - Identify rollback criteria

2. **After implementing:**
   - Run unit tests
   - Measure file size
   - Run quick perplexity (20 chunks)
   - Run speed benchmark (100 tokens)

3. **Go/No-Go decision:**
   - Size: Must not increase (unless quality gain > 1 PPL)
   - Quality: Must not degrade > 0.3 PPL
   - Speed: Must not slow down > 20%

4. **Documentation:**
   - Record all measurements
   - Keep before/after code diffs
   - Maintain changelog

---

## Phase 6: Implementation Order

### Tier 1: Must Do (Foundation)
| Step | Description | Expected Impact |
|------|-------------|-----------------|
| 0.1 | Baseline measurement | None (measurement only) |
| 1.1 | FP16 scale | -2 bytes/block, no quality impact |
| 2.1 | Unit tests | None (testing only) |

### Tier 2: Should Do (Optimization)
| Step | Description | Expected Impact |
|------|-------------|-----------------|
| 3.1 | Profile hotspots | None (analysis only) |
| 3.2 | Optimize extraction | Speed improvement |
| 3.3 | Outlier optimization | Speed improvement |

### Tier 3: Could Do (Experimental)
| Step | Description | Expected Impact |
|------|-------------|-----------------|
| 1.2 | Implicit indices | -6 bytes/block, minor quality risk |
| 4.2 | Hybrid outlier format | -6 bytes/block, HIGH quality risk |
| 4.3 | Static outlier positions | -6 bytes/block, medium quality risk |

### Tier 4: Deferred
| Step | Description | Reason |
|------|-------------|--------|
| 4.1 | 128-block size | Breaks Q8_K compatibility |
| 3.4 | Fused matmul | Complex, needs careful verification |

---

## Changelog

| Date | Step | Change | Size | PPL | Speed | Status |
|------|------|--------|------|-----|-------|--------|
| 2025-12-11 | 0.1 | Baseline Q3_K_M | 1023.52 MiB | 22.78 | ~56 tok/s | ✅ Done |
| 2025-12-11 | 0.1 | Baseline Q3_HIFI (original) | 1044.31 MiB | - | ~0.85 tok/s | ✅ Done |
| 2025-12-11 | 1.1 | FP16 scale (float d → ggml_fp16_t d) | -2 bytes/block | - | - | ✅ Done |
| 2025-12-11 | 1.1b | uint8 outlier indices (uint16 → uint8) | -6 bytes/block | - | - | ✅ Done |
| 2025-12-11 | 3.1 | AVX2 vec_dot (continuous 3-bit) | - | 21.91 | 10 tok/s | ✅ Done |
| 2025-12-11 | 3.2 | Split ql/qh format (qs[96] → ql[64]+qh[32]) | same | 21.91 | 9 tok/s | ✅ Done |
| 2025-12-11 | - | **Final Q3_HIFI v7** | **987.37 MiB** | **21.91** | **9 tok/s** | ✅ Current |

### Key Insights from Format Change (3.2):
- **Quantization 2x faster**: 26s → 11s (simpler bit packing)
- **Speed unchanged**: Still ~9-10 tok/s (extraction still scalar)
- **Foundation for SIMD**: Split layout enables future pure-SIMD extraction
- **Quality preserved**: PPL unchanged at 21.91

---

## Notes

- Always quantize fresh models after format changes
- Keep reference (generic) implementations working
- GPU shaders must be updated in sync with CPU code
- Test on multiple models if possible (not just Qwen3-1.7B)

---

## Analysis: Why Q3_HIFI is 6x Slower than Q3_K (Updated 2025-12-11)

### ❌ NOT the cause (contrary to some analysis):
- ~~vec_dot kernel not registered~~ → **Actually IS registered** in `ggml-cpu.c`
- ~~Falling back to generic dequant+matmul~~ → **Actually uses AVX2 vec_dot**
- ~~Wrong function optimized~~ → **Correct function is being called**
- ~~Continuous 3-bit packing~~ → **Now using split ql/qh layout**

### ✅ ACTUAL root cause (current):
**Extraction is still scalar before SIMD dot product**

| Aspect | Q3_K (fast) | Q3_HIFI v7 (slow) |
|--------|-------------|-------------------|
| Layout | Split `ql[64]` + `qh[32]` | Split `ql[64]` + `qh[32]` ✅ |
| Bit extraction | **Pure SIMD shuffles** | Scalar loop, then SIMD ❌ |
| SIMD friendliness | Full pipeline | Broken by extraction |
| Outlier handling | N/A | 6 FP16 corrections per block |

### What we've achieved:
1. ✅ **Split ql/qh layout** - Foundation for SIMD (Step 3.2)
2. ✅ **Quantization 2x faster** - Simpler bit packing
3. ✅ **Quality preserved** - PPL 21.91 (better than Q3_K's 22.78)
4. ⚠️ **Speed still 6x slower** - Extraction not yet SIMD

### Remaining bottleneck:
```c
// Current: Extract 256 values one at a time, then SIMD dot product
for (int i = 0; i < 256; i += 8) {
    uint8_t ql0 = ql[ql_idx];
    uint8_t qh_byte = qh[qh_idx];
    q3[i+0] = ((ql0 >> 0) & 0x03) | (((qh_byte >> 0) & 1) << 2) - 4;
    // ... still scalar extraction
}
```

### Path to Q3_K-level speed:
1. **Pure SIMD extraction** - Use shuffle/blend like Q3_K (complex)
2. **Or: Pre-extract to LUT** - Trade memory for speed
3. **Pre-zero outliers** (Step 3.3) - Eliminates subtract ops

---

## Quick Reference: Current vs Target

```
Original Q3_HIFI v1 (124 bytes/256 weights = 3.875 BPW):
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ float d (4B) │ qs[96] (96B) │ idx[6] (12B uint16) │ vals[6] (12B FP16) │
└─────────────────────────────────────────────────────────────────────────────────────────┘

Previous Q3_HIFI v4 (116 bytes, continuous 3-bit packing):
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ fp16 d (2B) │ qs[96] (96B) │ idx[6] (6B uint8) │ vals[6] (12B FP16) │
└─────────────────────────────────────────────────────────────────────────────────────────┘

Current Q3_HIFI v7 (116 bytes/256 weights = 3.625 BPW): ✅ ACHIEVED
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ fp16 d (2B) │ ql[64] (64B) │ qh[32] (32B) │ idx[6] (6B) │ vals[6] (12B) │
└─────────────────────────────────────────────────────────────────────────────────────────┘
(split ql/qh layout for SIMD-friendly extraction)

Target Q3_HIFI v8 (107 bytes/256 weights = 3.34 BPW): 🎯 NEXT
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ fp16 d (2B) │ ql[64] (64B) │ qh[32] (32B) │ idx[6] (6B) │ codes[3] (3B) │
└─────────────────────────────────────────────────────────────────────────────────────────┘
(outlier vals replaced with 4-bit codebook indices - saves 9 bytes!)

Q3_K reference (110 bytes/256 weights = 3.44 BPW):
┌────────────────────────────────────────────────────────────────────────────────┐
│ fp16 d (2B) │ hmask[32] (32B) │ qs[64] (64B) │ scales[12] (12B) │
└────────────────────────────────────────────────────────────────────────────────┘
```

