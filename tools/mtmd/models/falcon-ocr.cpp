#include "models.h"

// Important: Falcon OCR model does not actually have a vision encoder,
// the image patches are projected directly to the text embedding space and fed to the text encoder.
// we only use the mmproj to store the image projector weights and preprocessor config

ggml_cgraph * clip_graph_falcon_ocr::build() {
    ggml_tensor * inp_raw = build_inp_raw();

    const int ps = patch_size;
    const int pw = img.nx / ps;
    const int ph = img.ny / ps;
    const int n_patch = pw * ph;

    ggml_tensor * proj_w = ggml_reshape_4d(ctx0, model.mm_0_w, ps, ps, 3, n_embd);

    ggml_tensor * cur = ggml_conv_2d(ctx0, proj_w, inp_raw, ps, ps, 0, 0, 1, 1);

    // conv2d output [OW, OH, OC, 1] -> [n_embd, n_patch]
    cur = ggml_reshape_2d(ctx0, cur, n_patch, n_embd);
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    // prepend the BOI tokens (multiple tokens)
    cur = ggml_concat(ctx0, model.mm_img_begin, cur, 1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
