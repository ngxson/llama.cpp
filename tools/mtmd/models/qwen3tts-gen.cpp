#include "models.h"

ggml_cgraph * clip_graph_qwen3tts_gen::build() {
    ggml_tensor * h_state = build_inp_raw(1);
    h_state = ggml_reshape_1d(ctx0, h_state, h_state->ne[0]);
    cb(h_state, "inp_h_state", -1);

    // inp_code0: sampled codebook-0 id from the talker's codec_head
    ggml_tensor * code0_embd; // [n_embd, 1]
    {
        ggml_tensor * code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_name(code0, "inp_code0");
        ggml_set_input(code0);

        ggml_tensor * code0_embd = ggml_get_rows(ctx0, model.gen_code_out_embd_w, code0);
        code0_embd = ggml_reshape_1d(ctx0, code0_embd, code0_embd->ne[0]);
        cb(code0_embd, "code0_embd", -1);
    }

    // TODO: code_predictor transformer block (5 layers) + per-codebook
    // sampling loop; placeholder combination so the graph has a single
    // well-defined output tensor until that lands
    ggml_tensor * cur = ggml_add(ctx0, h_state, code0_embd);
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[0], 1);
    cb(cur, "gen_audio_out", -1);

    ggml_build_forward_expand(gf, cur);
    return gf;
}
