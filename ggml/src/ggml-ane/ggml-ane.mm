// ggml-ane.mm — Apple Neural Engine backend for GGML
//
// MUL_MAT uses the "dynamic weight trick" from simple_demo.m:
//   activation (src1) and weight (src0) are packed side-by-side in one IOSurface,
//   a MIL program slices them, reshapes, and calls matmul on the ANE.
//
// All three tensor dimensions are padded up to the hardware-required minimums before
// compiling a kernel.  Empirically, the ANE needs:
//   IC  (K) ≥ ANE_MIN_IC  = 256  (multiples of ANE_ALIGN = 64)
//   OC  (N) ≥ ANE_MIN_OC  = 128  (multiples of ANE_ALIGN = 64)
//   SEQ (M) ≥ ANE_MIN_SEQ =  64  (multiples of ANE_ALIGN = 64)
//
// These match the smallest dimensions that work in simple_demo.m (IC=256, OC=128,
// SEQ=64).  Smaller actual dimensions are zero-padded; only the valid [K×N×M] slice
// is written/read.
//
// IOSurface input layout  [IC_p rows × (SEQ_p + OC_p) fp16 columns]:
//   row ic: [ src1[s·K+ic for s=0..M-1] | 0 for s=M..SEQ_p-1 |
//             src0[oc·K+ic for oc=0..N-1] | 0 for oc=N..OC_p-1 ]
//
// IOSurface output layout  [OC_p × SEQ_p fp16, NCHW [1,OC_p,1,SEQ_p]]:
//   out[oc·SEQ_p + s]  →  dst_f[s·N + oc]   (valid: oc < N, s < M)

#include "ggml-ane.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <cstring>

// ============================================================
// ANE dimension constraints (matching PoC / hardware minimums)
// ============================================================

static constexpr int64_t ANE_ALIGN   =  64;  // all dims must be multiples of this
static constexpr int64_t ANE_MIN_IC  = 256;  // minimum inner dimension (K)
static constexpr int64_t ANE_MIN_OC  = 128;  // minimum output channels (N)
static constexpr int64_t ANE_MIN_SEQ =  64;  // minimum sequence length (M)

static int64_t ane_pad_ic (int64_t x) {
    int64_t r = (x + ANE_ALIGN - 1) / ANE_ALIGN * ANE_ALIGN;
    return std::max(r, ANE_MIN_IC);
}
static int64_t ane_pad_oc (int64_t x) {
    int64_t r = (x + ANE_ALIGN - 1) / ANE_ALIGN * ANE_ALIGN;
    return std::max(r, ANE_MIN_OC);
}
static int64_t ane_pad_seq(int64_t x) {
    int64_t r = (x + ANE_ALIGN - 1) / ANE_ALIGN * ANE_ALIGN;
    return std::max(r, ANE_MIN_SEQ);
}

// ============================================================
// Kernel cache key  (keyed on padded dims to maximise reuse)
// ============================================================

struct ane_kern_key {
    int64_t IC, OC, SEQ;
    bool operator==(const ane_kern_key & o) const {
        return IC == o.IC && OC == o.OC && SEQ == o.SEQ;
    }
};

struct ane_kern_key_hash {
    size_t operator()(const ane_kern_key & k) const {
        size_t h = std::hash<int64_t>()(k.IC);
        h ^= std::hash<int64_t>()(k.OC)  + 0x9e3779b9u + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>()(k.SEQ) + 0x9e3779b9u + (h << 6) + (h >> 2);
        return h;
    }
};

// ============================================================
// Compiled ANE kernel (one per unique padded IC×OC×SEQ)
// ============================================================

struct ane_kern {
    void *       model;    // __bridge_retained _ANEInMemoryModel
    void *       request;  // __bridge_retained _ANERequest
    IOSurfaceRef ioIn;     // [IC_p × (SEQ_p + OC_p)] fp16  (activation + weight packed)
    IOSurfaceRef ioOut;    // [OC_p × SEQ_p] fp16
    std::string  tmpDir;
    int64_t      IC, OC, SEQ;   // padded dimensions
};

// ============================================================
// ANE private-framework class handles (loaded once)
// ============================================================

static Class g_Desc = nil;
static Class g_IMM  = nil;
static Class g_AR   = nil;
static Class g_AIO  = nil;

static bool ane_load_classes(void) {
    if (g_Desc) {
        return true;
    }
    void *h = dlopen(
        "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
        RTLD_NOW);
    if (!h) {
        GGML_LOG_DEBUG("ggml-ane: AppleNeuralEngine.framework not found\n");
        return false;
    }
    g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_IMM  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR   = NSClassFromString(@"_ANERequest");
    g_AIO  = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_Desc || !g_IMM || !g_AR || !g_AIO) {
        GGML_LOG_DEBUG("ggml-ane: failed to locate ANE private classes\n");
        return false;
    }
    return true;
}

static bool ane_is_available(void) {
    static bool s_checked   = false;
    static bool s_available = false;
    if (!s_checked) {
        @autoreleasepool {
            s_available = ane_load_classes();
        }
        s_checked = true;
    }
    return s_available;
}

// ============================================================
// IOSurface helper
// ============================================================

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth          : @(bytes),
        (id)kIOSurfaceHeight         : @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow    : @(bytes),
        (id)kIOSurfaceAllocSize      : @(bytes),
        (id)kIOSurfacePixelFormat    : @0,
    });
}

// ============================================================
// MIL program generation (from simple_demo.m, parameterised)
// ============================================================
//
// Input:  [1, IC, 1, SEQ+OC]  — activation and weight packed row-wise
// Output: [1, OC, 1, SEQ]

static NSString * gen_mil(int64_t IC, int64_t OC, int64_t SEQ) {
    const int64_t sp = SEQ + OC;
    NSMutableString *m = [NSMutableString string];

    [m appendString:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
        "{\n"];

    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %lld, 1, %lld]> x) {\n", IC, sp];

    // slice activation: cols 0..SEQ-1 → [1, IC, 1, SEQ]
    [m appendString:
        @"        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:
        @"        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,%lld,1,%lld])];\n", IC, SEQ];
    [m appendFormat:
        @"        tensor<fp16, [1,%lld,1,%lld]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n", IC, SEQ];

    // slice weight: cols SEQ..SEQ+OC-1 → [1, IC, 1, OC]
    [m appendFormat:
        @"        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,%lld])];\n", SEQ];
    [m appendFormat:
        @"        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,%lld,1,%lld])];\n", IC, OC];
    [m appendFormat:
        @"        tensor<fp16, [1,%lld,1,%lld]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n", IC, OC];

    // reshape + transpose activation: [1,IC,1,SEQ] → [1,1,IC,SEQ] → [1,1,SEQ,IC]
    [m appendFormat:
        @"        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,%lld,%lld])];\n", IC, SEQ];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%lld,%lld]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n", IC, SEQ];
    [m appendString:
        @"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%lld,%lld]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n", SEQ, IC];

    // reshape weight: [1,IC,1,OC] → [1,1,IC,OC]
    [m appendFormat:
        @"        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,%lld,%lld])];\n", IC, OC];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%lld,%lld]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n", IC, OC];

    // matmul: [1,1,SEQ,IC] @ [1,1,IC,OC] → [1,1,SEQ,OC]
    [m appendString:
        @"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendFormat:
        @"        tensor<fp16, [1,1,%lld,%lld]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", SEQ, OC];

    // transpose back + reshape: [1,1,SEQ,OC] → [1,1,OC,SEQ] → [1,OC,1,SEQ]
    [m appendFormat:
        @"        tensor<fp16, [1,1,%lld,%lld]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n", OC, SEQ];
    [m appendFormat:
        @"        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,%lld,1,%lld])];\n", OC, SEQ];
    [m appendFormat:
        @"        tensor<fp16, [1,%lld,1,%lld]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n", OC, SEQ];

    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

// ============================================================
// Compile + load one ANE kernel
// ============================================================

static ane_kern * ane_compile(int64_t IC, int64_t OC, int64_t SEQ) {
    NSString *mil = gen_mil(IC, OC, SEQ);
    NSData   *md  = [mil dataUsingEncoding:NSUTF8StringEncoding];

    id desc = ((id(*)(Class, SEL, id, id, id))objc_msgSend)(
        g_Desc, @selector(modelWithMILText:weights:optionsPlist:), md, @{}, nil);
    if (!desc) {
        GGML_LOG_DEBUG("ggml-ane: compile: descriptor=NULL (IC=%lld OC=%lld SEQ=%lld)\n", IC, OC, SEQ);
        return nullptr;
    }

    id mdl = ((id(*)(Class, SEL, id))objc_msgSend)(
        g_IMM, @selector(inMemoryModelWithDescriptor:), desc);

    // ANE looks up compiled artefacts by hex ID under /tmp/<hexId>/
    NSString *hx = ((id(*)(id, SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
        withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];

    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        GGML_LOG_DEBUG("ggml-ane: compile failed: %s\n", e ? [[e description] UTF8String] : "?");
        return nullptr;
    }

    ok = ((BOOL(*)(id, SEL, unsigned int, id, NSError **))objc_msgSend)(
        mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
    if (!ok) {
        GGML_LOG_DEBUG("ggml-ane: load failed: %s\n", e ? [[e description] UTF8String] : "?");
        return nullptr;
    }

    const int64_t sp = SEQ + OC;
    IOSurfaceRef ioIn  = make_surface((size_t)(IC * sp  * (int64_t)sizeof(_Float16)));
    IOSurfaceRef ioOut = make_surface((size_t)(OC * SEQ * (int64_t)sizeof(_Float16)));

    id wI  = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioIn);
    id wO  = ((id(*)(Class, SEL, IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), ioOut);
    id req = ((id(*)(Class, SEL, id, id, id, id, id, id, id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0);

    ane_kern *k = new ane_kern;
    k->model   = (__bridge_retained void *)mdl;
    k->request = (__bridge_retained void *)req;
    k->ioIn    = ioIn;
    k->ioOut   = ioOut;
    k->tmpDir  = std::string([td UTF8String]);
    k->IC = IC;  k->OC = OC;  k->SEQ = SEQ;
    return k;
}

static void ane_run(ane_kern * k) {
    id mdl = (__bridge id)k->model;
    id req = (__bridge id)k->request;
    NSError *e = nil;
    BOOL ok = ((BOOL(*)(id, SEL, unsigned int, id, id, NSError **))objc_msgSend)(
        mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
    if (!ok || e) {
        GGML_LOG_DEBUG("ggml-ane: eval error: %s\n", e ? [[e description] UTF8String] : "evaluate returned NO");
    }
}

static void ane_kern_free(ane_kern * k) {
    if (!k) {
        return;
    }
    @autoreleasepool {
        id mdl = (__bridge id)k->model;
        NSError *e = nil;
        ((BOOL(*)(id, SEL, unsigned int, NSError **))objc_msgSend)(
            mdl, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(k->ioIn);
        CFRelease(k->ioOut);
        [[NSFileManager defaultManager]
            removeItemAtPath:[NSString stringWithUTF8String:k->tmpDir.c_str()]
            error:nil];
        CFRelease(k->model);
        CFRelease(k->request);
    }
    delete k;
}

// ============================================================
// Backend context
// ============================================================

struct ggml_backend_ane_context {
    std::unordered_map<ane_kern_key, ane_kern *, ane_kern_key_hash> kernels;

    ~ggml_backend_ane_context() {
        for (auto & kv : kernels) {
            ane_kern_free(kv.second);
        }
    }

    ane_kern * get_or_compile(int64_t IC_p, int64_t OC_p, int64_t SEQ_p) {
        ane_kern_key key{IC_p, OC_p, SEQ_p};
        auto it = kernels.find(key);
        if (it != kernels.end()) {
            return it->second;
        }
        @autoreleasepool {
            ane_kern *k = ane_compile(IC_p, OC_p, SEQ_p);
            kernels[key] = k;
            return k;
        }
    }
};

// ============================================================
// MUL_MAT
// ============================================================

static void ggml_backend_ane_mul_mat(ggml_backend_ane_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // weight [K, N, ...]
    const struct ggml_tensor * src1 = dst->src[1];  // input  [K, M, ...]

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    // Pad to hardware minimums; identical padded shapes share one compiled kernel
    const int64_t IC_p  = ane_pad_ic(K);
    const int64_t OC_p  = ane_pad_oc(N);
    const int64_t SEQ_p = ane_pad_seq(M);

    const int64_t ne2      = dst->ne[2];
    const int64_t ne3      = dst->ne[3];
    const int64_t src0_ne2 = src0->ne[2];
    const int64_t src0_ne3 = src0->ne[3];
    // GQA-style broadcast: use integer division i02=b2/r2 (not modulo), matching ggml CPU / ZenDNN.
    // Guard against src0_ne[i] == 0 (should never happen but prevents UB from integer div-by-zero).
    const int64_t r2 = (src0_ne2 > 0) ? ne2 / src0_ne2 : 1;
    const int64_t r3 = (src0_ne3 > 0) ? ne3 / src0_ne3 : 1;

    ane_kern *k = ctx->get_or_compile(IC_p, OC_p, SEQ_p);
    if (!k) {
        memset(dst->data, 0, ggml_nbytes(dst));
        return;
    }

    const int64_t sp = SEQ_p + OC_p;   // IOSurface row width

    @autoreleasepool {
        for (int64_t b3 = 0; b3 < ne3; b3++) {
            for (int64_t b2 = 0; b2 < ne2; b2++) {
                const int64_t s0b2 = b2 / r2;
                const int64_t s0b3 = b3 / r3;
                // Use nb[] byte-strides (not ne[]-reconstructed element counts) so that
                // GQA-style broadcast with odd bs[0] (e.g. 3) is computed correctly.
                // Keep as const char * to avoid strict-aliasing UB when reinterpreting
                // the weight pointer as _Float16 * below.
                const char *s0_bytes = (const char *)src0->data + s0b2 * src0->nb[2] + s0b3 * src0->nb[3];
                const float *s1      = (const float *)((const char *)src1->data + b2 * src1->nb[2] + b3 * src1->nb[3]);
                float       *dt      = (float *)      ((      char *)dst->data  + b2 * dst->nb[2]  + b3 * dst->nb[3]);

                // Pack: IC_p rows × sp fp16 columns
                // Row ic = [activation cols 0..SEQ_p-1] [weight cols 0..OC_p-1]
                IOSurfaceLock(k->ioIn, 0, NULL);
                _Float16 *buf = (_Float16 *)IOSurfaceGetBaseAddress(k->ioIn);
                memset(buf, 0, (size_t)(IC_p * sp) * sizeof(_Float16));
                if (src0->type == GGML_TYPE_F16) {
                    // F16 weights — direct copy, no conversion needed
                    const _Float16 *s0h = (const _Float16 *)s0_bytes;
                    for (int64_t ic = 0; ic < K; ic++) {
                        for (int64_t s = 0; s < M; s++) {
                            buf[ic * sp + s] = (_Float16)s1[s * K + ic];
                        }
                        for (int64_t oc = 0; oc < N; oc++) {
                            buf[ic * sp + SEQ_p + oc] = s0h[oc * K + ic];
                        }
                    }
                } else {
                    // F32 weights (only reachable with GGML_ANE_ALLOW_F32_WEIGHTS)
                    const float *s0 = (const float *)s0_bytes;
                    for (int64_t ic = 0; ic < K; ic++) {
                        for (int64_t s = 0; s < M; s++) {
                            buf[ic * sp + s] = (_Float16)s1[s * K + ic];
                        }
                        for (int64_t oc = 0; oc < N; oc++) {
                            buf[ic * sp + SEQ_p + oc] = (_Float16)s0[oc * K + ic];
                        }
                    }
                }
                IOSurfaceUnlock(k->ioIn, 0, NULL);

                ane_run(k);

                // Unpack: ANE output [1, OC_p, 1, SEQ_p] linear as out[oc*SEQ_p+s].
                // Use dst's own nb[] strides rather than assuming a flat [N, M] layout,
                // in case dst is a non-contiguous view (supports_op doesn't check dst).
                IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
                const _Float16 *out = (const _Float16 *)IOSurfaceGetBaseAddress(k->ioOut);
                for (int64_t oc = 0; oc < N; oc++) {
                    for (int64_t s = 0; s < M; s++) {
                        float *elem = (float *)((char *)dt + s * dst->nb[1] + oc * dst->nb[0]);
                        *elem = (float)out[oc * SEQ_p + s];
                    }
                }
                IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
            }
        }
    }
}

// ============================================================
// Backend interface
// ============================================================

static const char * ggml_backend_ane_get_name(ggml_backend_t backend) {
    return "ANE";
    GGML_UNUSED(backend);
}

static void ggml_backend_ane_free(ggml_backend_t backend) {
    ggml_backend_ane_context *ctx = (ggml_backend_ane_context *)backend->context;
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_ane_graph_compute(ggml_backend_t backend, struct ggml_cgraph *cgraph) {
    ggml_backend_ane_context *ctx = (ggml_backend_ane_context *)backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor *node = cgraph->nodes[i];
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_ane_mul_mat(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
    return GGML_STATUS_SUCCESS;
    GGML_UNUSED(backend);
}

static struct ggml_backend_i ane_backend_i = {
    /* .get_name                = */ ggml_backend_ane_get_name,
    /* .free                    = */ ggml_backend_ane_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_ane_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_ane_guid(void) {
    static ggml_guid guid = {
        0xa1, 0x3e, 0x7b, 0x52, 0xc4, 0x9f, 0x4d, 0x08,
        0xbe, 0x71, 0x25, 0x6a, 0x0c, 0xd8, 0xe3, 0xf0,
    };
    return &guid;
}

ggml_backend_t ggml_backend_ane_init(void) {
    ggml_backend_ane_context *ctx = new ggml_backend_ane_context;
    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_ane_guid(),
        /* .iface   = */ ane_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_ane_reg(), 0),
        /* .context = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_ane(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_ane_guid());
}

// ============================================================
// Device interface
// ============================================================

static const char * ggml_backend_ane_device_get_name(ggml_backend_dev_t dev) {
    return "ANE";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_ane_device_get_description(ggml_backend_dev_t dev) {
    return "Apple Neural Engine";
    GGML_UNUSED(dev);
}

static void ggml_backend_ane_device_get_memory(ggml_backend_dev_t dev, size_t *free, size_t *total) {
    // ANE uses unified memory — report system RAM via Mach VM stats.
    vm_statistics64_data_t vmstat;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          (host_info64_t)&vmstat, &count) == KERN_SUCCESS) {
        const size_t page = vm_page_size;
        *total = (size_t)(vmstat.free_count + vmstat.active_count +
                          vmstat.inactive_count + vmstat.wire_count) * page;
        *free  = (size_t)vmstat.free_count * page;
    } else {
        *free = 0;  *total = 0;
    }
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_ane_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_ane_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
    props->name        = ggml_backend_ane_device_get_name(dev);
    props->description = ggml_backend_ane_device_get_description(dev);
    props->type        = ggml_backend_ane_device_get_type(dev);
    ggml_backend_ane_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_ane_device_init_backend(ggml_backend_dev_t dev, const char *params) {
    return ggml_backend_ane_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_ane_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_ane_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev, void *ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_ane_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor *op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            // ANE is a native fp16 accelerator.
            // F32 src0 is supported but requires a lossy F32→F16 conversion per call,
            // which accumulates enough error across layers to degrade model accuracy.
            // Define GGML_ANE_ALLOW_F32_WEIGHTS to opt into F32 weight support anyway.
            const bool src0_ok =
#ifdef GGML_ANE_ALLOW_F32_WEIGHTS
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
#else
                (op->src[0]->type == GGML_TYPE_F16);
#endif
            return ane_is_available()            &&
                   src0_ok                       &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0])    &&
                   ggml_is_contiguous(op->src[1]);
        }

        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_ane_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_ane_device_i = {
    /* .get_name             = */ ggml_backend_ane_device_get_name,
    /* .get_description      = */ ggml_backend_ane_device_get_description,
    /* .get_memory           = */ ggml_backend_ane_device_get_memory,
    /* .get_type             = */ ggml_backend_ane_device_get_type,
    /* .get_props            = */ ggml_backend_ane_device_get_props,
    /* .init_backend         = */ ggml_backend_ane_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_ane_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_ane_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_ane_device_supports_op,
    /* .supports_buft        = */ ggml_backend_ane_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// ============================================================
// Backend registration interface
// ============================================================

static const char * ggml_backend_ane_reg_get_name(ggml_backend_reg_t reg) {
    return "ANE";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_ane_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_ane_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_ane_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &dev;
    GGML_UNUSED(reg);
}

static void * ggml_backend_ane_get_proc_address(ggml_backend_reg_t reg, const char *name) {
    return NULL;
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_ane_reg_i = {
    /* .get_name         = */ ggml_backend_ane_reg_get_name,
    /* .get_device_count = */ ggml_backend_ane_reg_get_device_count,
    /* .get_device       = */ ggml_backend_ane_reg_get_device,
    /* .get_proc_address = */ ggml_backend_ane_get_proc_address,
};

ggml_backend_reg_t ggml_backend_ane_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_ane_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_ane_reg)
