#include "models.h"

#include <string>

// on-device sampling: top-k, top-p, then a random draw
ggml_tensor * clip_graph_qwen3tts_gen::do_sampling(ggml_tensor * logits, ggml_tensor * inp_rand, int top_k, float top_p) const {
    logits = ggml_reshape_1d(ctx0, logits, ggml_nelements(logits));
    const int64_t n_vocab = logits->ne[0];

    // sort a's rows by idx
    auto sort_by = [this](ggml_tensor * a, ggml_tensor * idx) {
        ggml_tensor * a2d = ggml_reshape_2d(ctx0, a, 1, a->ne[0]);
        return ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, a2d, idx), idx->ne[0]);
    };

    ggml_tensor * cur        = logits;
    ggml_tensor * candidates = nullptr; // maps row index back to vocab id

    if (top_k > 0 && top_k < n_vocab) {
        ggml_tensor * idx = ggml_top_k(ctx0, cur, top_k);
        candidates = idx;
        cur        = sort_by(cur, idx);
        cb(cur, "sample_top_k_logits", -1);
    }

    if (top_p < 1.0f) {
        ggml_tensor * sorted_idx    = ggml_argsort(ctx0, cur, GGML_SORT_ORDER_DESC);
        ggml_tensor * sorted_logits = sort_by(cur, sorted_idx);
        candidates = candidates ? sort_by(candidates, sorted_idx) : sorted_idx;

        ggml_tensor * probs = ggml_soft_max(ctx0, sorted_logits);
        ggml_tensor * cdf   = ggml_cumsum(ctx0, probs);

        // keep_mask[i] = 1 once cdf[i] crosses top_p
        ggml_tensor * cdf_scaled = ggml_scale_bias(ctx0, cdf, -1.0f, top_p);
        ggml_tensor * keep_mask  = ggml_step(ctx0, cdf_scaled);
        ggml_tensor * idxf       = ggml_sum(ctx0, keep_mask);
        idxf = ggml_clamp(ctx0, idxf, 0.0f, (float) keep_mask->ne[0] - 1);
        ggml_tensor * ones = ggml_scale_bias(ctx0, idxf, 0.0f, 1.0f);

        // top-p must include the crossing element, so force it to 1
        ggml_tensor * keep_mask_2d = ggml_reshape_2d(ctx0, keep_mask, 1, keep_mask->ne[0]);
        keep_mask_2d = ggml_set_rows(ctx0, keep_mask_2d, ones, ggml_cast(ctx0, idxf, GGML_TYPE_I32));
        keep_mask    = ggml_reshape_1d(ctx0, keep_mask_2d, keep_mask->ne[0]);

        // log(1) = 0 (keep), log(0) = -inf (drop)
        ggml_tensor * bias = ggml_log(ctx0, keep_mask);
        cur = ggml_add(ctx0, sorted_logits, bias);
        cb(cur, "sample_top_p_logits", -1);
    }

    // draw one token: find where the cdf crosses inp_rand
    ggml_tensor * probs  = ggml_soft_max(ctx0, cur);
    ggml_tensor * cumsum = ggml_cumsum(ctx0, probs);

    ggml_tensor * diff       = ggml_sub(ctx0, cumsum, inp_rand);
    ggml_tensor * cross_mask = ggml_step(ctx0, diff);
    ggml_tensor * idxf       = ggml_sum(ctx0, cross_mask);
    ggml_tensor * idx        = ggml_cast(ctx0, ggml_scale_bias(ctx0, idxf, -1.0f, (float) cross_mask->ne[0]), GGML_TYPE_I32);

    if (candidates) {
        ggml_tensor * cand_2d = ggml_reshape_2d(ctx0, candidates, 1, candidates->ne[0]);
        idx = ggml_get_rows(ctx0, cand_2d, idx);
    }
    cb(idx, "sample_token_id", -1);

    return idx;
}

// returns a new cache with row row_idx set to value
ggml_tensor * clip_graph_qwen3tts_gen::cache_set(ggml_tensor * cache, int row_idx, ggml_tensor * value) const {
    const int64_t n_embd  = cache->ne[0];
    const int64_t n_cache = cache->ne[1];
    GGML_ASSERT(row_idx >= 0 && row_idx < n_cache);

    // append value as the last row, then gather it back into place
    ggml_tensor * value_2d  = ggml_reshape_2d(ctx0, value, n_embd, 1);
    ggml_tensor * cache_ext = ggml_concat(ctx0, cache, value_2d, 1); // [n_embd, n_cache + 1]

    ggml_tensor * idx = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) n_cache, 1.0f), GGML_TYPE_I32);

    ggml_tensor * pos      = const_i32(cache, (float) row_idx);
    ggml_tensor * new_slot = const_i32(cache, (float) n_cache);

    // idx[row_idx] = n_cache, so that row now gathers the appended value
    ggml_tensor * idx_2d = ggml_reshape_2d(ctx0, idx, 1, n_cache);
    idx_2d = ggml_set_rows(ctx0, idx_2d, ggml_reshape_2d(ctx0, new_slot, 1, 1), pos);
    idx    = ggml_reshape_1d(ctx0, idx_2d, n_cache);

    ggml_tensor * result = ggml_get_rows(ctx0, cache_ext, idx);
    cb(result, "cache_set_out", -1);
    return result;
}

// builds a const i32 value with no host upload: view any f32 tensor,
// scale it to 0, add the value, cast to i32
ggml_tensor * clip_graph_qwen3tts_gen::const_i32(ggml_tensor * anchor, float value) const {
    ggml_tensor * v = ggml_view_1d(ctx0, anchor, 1, 0);
    return ggml_cast(ctx0, ggml_scale_bias(ctx0, v, 0.0f, value), GGML_TYPE_I32);
}

// causal keep-mask row for a query at position pos, window size n_kv_pad
ggml_tensor * clip_graph_qwen3tts_gen::causal_mask_row(int64_t n_kv_pad, int pos) const {
    ggml_tensor * ones = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv_pad, n_kv_pad), 1.0f);
    ggml_tensor * keep = ggml_tri(ctx0, ones, GGML_TRI_TYPE_LOWER_DIAG);
    ggml_tensor * row  = ggml_view_1d(ctx0, keep, n_kv_pad, (size_t) pos * keep->nb[1]);
    ggml_tensor * mask = ggml_log(ctx0, row); // 0 = keep, -inf = masked
    return ggml_reshape_4d(ctx0, mask, n_kv_pad, 1, 1, 1);
}

// talker hidden size -> predictor hidden size (small_to_mtp_projection)
ggml_tensor * clip_graph_qwen3tts_gen::project_in(ggml_tensor * cur) const {
    if (!model.gen_code_proj_in_w) {
        return cur;
    }
    cur = ggml_mul_mat(ctx0, model.gen_code_proj_in_w, cur);
    if (model.gen_code_proj_in_b) {
        cur = ggml_add(ctx0, cur, model.gen_code_proj_in_b);
    }
    return cur;
}

// one transformer layer at a single new position pos; writes k/v into
// k_cache_layer/v_cache_layer at row pos
ggml_tensor * clip_graph_qwen3tts_gen::layer_forward(
        ggml_tensor * cur,
        const clip_layer & layer,
        ggml_tensor * inp_pos,
        ggml_tensor * kq_mask,
        ggml_tensor *& k_cache_layer,
        ggml_tensor *& v_cache_layer,
        int64_t n_kv_pad,
        int pos,
        int il) const {
    const int     n_head    = hparams.n_head;
    const int     n_head_kv = hparams.n_head_kv;
    const int64_t d_head    = layer.q_w->ne[1] / n_head; // real head_dim, not n_embd / n_head
    const float   kq_scale  = 1.0f / sqrtf((float) d_head);

    ggml_tensor * residual = cur;

    ggml_tensor * h = ggml_rms_norm(ctx0, cur, hparams.eps);
    h = ggml_mul(ctx0, h, layer.ln_1_w);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.q_w, h);
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.k_w, h);
    ggml_tensor * v = ggml_mul_mat(ctx0, layer.v_w, h);

    q = ggml_reshape_3d(ctx0, q, d_head, n_head, 1);
    k = ggml_reshape_3d(ctx0, k, d_head, n_head_kv, 1);

    q = ggml_rms_norm(ctx0, q, hparams.eps);
    q = ggml_mul(ctx0, q, layer.q_norm);
    k = ggml_rms_norm(ctx0, k, hparams.eps);
    k = ggml_mul(ctx0, k, layer.k_norm);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // write k/v into the cache at row pos, flat layout
    ggml_tensor * k_flat = ggml_reshape_1d(ctx0, k, d_head * n_head_kv);
    k_cache_layer = cache_set(k_cache_layer, pos, k_flat);
    v_cache_layer = cache_set(v_cache_layer, pos, v);

    ggml_tensor * q_cur = ggml_reshape_4d(ctx0, q, d_head, n_head, 1, 1);
    ggml_tensor * k_cur = ggml_reshape_4d(ctx0, k_cache_layer, d_head, n_head_kv, n_kv_pad, 1);
    ggml_tensor * v_cur = ggml_reshape_4d(ctx0, v_cache_layer, d_head, n_head_kv, n_kv_pad, 1);

    ggml_tensor * attn_out = build_attn(layer.o_w, layer.o_b, q_cur, k_cur, v_cur, kq_mask, kq_scale, il);

    cur = ggml_add(ctx0, residual, attn_out);

    ggml_tensor * h2 = ggml_rms_norm(ctx0, cur, hparams.eps);
    h2 = ggml_mul(ctx0, h2, layer.ln_2_w);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ff_gate_w, h2);
    ggml_tensor * up   = ggml_mul_mat(ctx0, layer.ff_up_w, h2);
    ggml_tensor * gu   = ggml_swiglu_split(ctx0, gate, up);
    ggml_tensor * down = ggml_mul_mat(ctx0, layer.ff_down_w, gu);

    return ggml_add(ctx0, cur, down);
}

// position 0: hidden bridge, no sampling, only seeds the k/v cache.
// position 1: embed(code0) via the talker's out_embd table, sample with
// lm_head[0], write out_code_cache[1].
void clip_graph_qwen3tts_gen::prefill(
        std::vector<ggml_tensor *> & k_cache,
        std::vector<ggml_tensor *> & v_cache,
        ggml_tensor *& out_code_cache,
        ggml_tensor * h_state,
        ggml_tensor * code0_embd,
        ggml_tensor * inp_rand,
        int top_k,
        float top_p) const {
    const int64_t n_kv_pad = k_cache[0]->ne[1];

    {
        ggml_tensor * cur     = project_in(h_state);
        ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, 0);
        ggml_tensor * inp_pos = const_i32(k_cache[0], 0.0f);
        for (size_t il = 0; il < model.layers.size(); il++) {
            cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, 0, (int) il);
        }
        // position 0's own output is not used further, it only seeded the cache
    }

    {
        ggml_tensor * cur     = project_in(code0_embd);
        ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, 1);
        ggml_tensor * inp_pos = const_i32(k_cache[0], 1.0f);
        for (size_t il = 0; il < model.layers.size(); il++) {
            cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, 1, (int) il);
        }

        cur = ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.gen_code_norm_w);

        ggml_tensor * head_w = model.gen_code_head_w;
        ggml_tensor * head_g = ggml_view_2d(ctx0, head_w, head_w->ne[0], head_w->ne[1], head_w->nb[1], 0); // lm_head[0]
        ggml_tensor * logits = ggml_mul_mat(ctx0, head_g, cur);

        ggml_tensor * sampled = do_sampling(logits, inp_rand, top_k, top_p);
        out_code_cache = cache_set(out_code_cache, 1, sampled);
    }
}

// one decode step of the 5-layer code_predictor.
// at step_idx g: read code from out_code_cache[g], embed it with codebook
// table g-1, write the new k/v at cache row g+1, sample with lm_head[g],
// write the result to out_code_cache[g+1].
//
// k_cache/v_cache: per layer, [d_head * n_head_kv, n_kv_pad].
// out_code_cache: [1, n_codes] I32. inp_rand: [1] F32 draw for this step.
// Create all input tensors in build(), not here.
// step_idx range: [1, n_acoustic - 1]. Returns the new out_code_cache.
ggml_tensor * clip_graph_qwen3tts_gen::step(
        std::vector<ggml_tensor *> & k_cache,
        std::vector<ggml_tensor *> & v_cache,
        ggml_tensor * out_code_cache,
        ggml_tensor * inp_rand,
        int step_idx,
        int top_k,
        float top_p) const {
    const int64_t n_acoustic = model.gen_code_head_w->ne[2];
    GGML_ASSERT(step_idx >= 1 && step_idx < n_acoustic);
    GGML_ASSERT(k_cache.size() == model.layers.size());
    GGML_ASSERT(v_cache.size() == model.layers.size());

    const int64_t n_kv_pad = k_cache[0]->ne[1];
    const int     pos      = step_idx + 1; // new cache row and RoPE position

    // embed the previous code through this step's codebook table
    // (out_code_cache has ne[0] == 1, so one row is already a single scalar)
    ggml_tensor * code_in = ggml_view_1d(ctx0, out_code_cache, 1, (size_t) step_idx * out_code_cache->nb[1]);

    ggml_tensor * embd_w = model.gen_code_embd_w; // [n_embd_talker, vocab, n_acoustic]
    ggml_tensor * embd_g = ggml_view_2d(ctx0, embd_w, embd_w->ne[0], embd_w->ne[1], embd_w->nb[1],
                                        (size_t) (step_idx - 1) * embd_w->nb[2]);
    ggml_tensor * cur = ggml_get_rows(ctx0, embd_g, code_in);
    cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);
    cb(cur, "step_embd_in", step_idx);

    cur = project_in(cur);
    cb(cur, "step_proj_in", step_idx);

    ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, pos);
    ggml_tensor * inp_pos = const_i32(k_cache[0], (float) pos);

    for (size_t il = 0; il < model.layers.size(); il++) {
        cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, pos, (int) il);
        cb(cur, "step_layer_out", (int) il);
    }

    // final norm, this step's lm_head, sample, write the result
    cur = ggml_rms_norm(ctx0, cur, hparams.eps);
    cur = ggml_mul(ctx0, cur, model.gen_code_norm_w);

    ggml_tensor * head_w = model.gen_code_head_w; // [n_embd_pred, vocab, n_acoustic]
    ggml_tensor * head_g = ggml_view_2d(ctx0, head_w, head_w->ne[0], head_w->ne[1], head_w->nb[1],
                                        (size_t) step_idx * head_w->nb[2]);
    ggml_tensor * logits = ggml_mul_mat(ctx0, head_g, cur);
    cb(logits, "step_logits", step_idx);

    ggml_tensor * sampled = do_sampling(logits, inp_rand, top_k, top_p);
    cb(sampled, "step_sampled", step_idx);

    return cache_set(out_code_cache, pos, sampled);
}

ggml_cgraph * clip_graph_qwen3tts_gen::build() {
    GGML_ASSERT(n_batch == 1); // this module only ever processes one frame at a time

    ggml_tensor * h_state = build_inp_raw(1);
    h_state = ggml_reshape_1d(ctx0, h_state, h_state->ne[0]);
    cb(h_state, "inp_h_state", -1);

    ggml_tensor * code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(code0, "inp_code0");
    ggml_set_input(code0);

    ggml_tensor * code0_embd = ggml_get_rows(ctx0, model.gen_code_out_embd_w, code0);
    code0_embd = ggml_reshape_1d(ctx0, code0_embd, code0_embd->ne[0]);
    cb(code0_embd, "code0_embd", -1);

    const int64_t n_acoustic = model.gen_code_head_w->ne[2]; // 15
    const int     n_codes    = (int) n_acoustic + 1;         // 16
    const int64_t n_kv_pad   = n_codes;
    const int     n_layer    = (int) model.layers.size();
    const int     n_head     = hparams.n_head;
    const int     n_head_kv  = hparams.n_head_kv;
    const int64_t d_head     = model.layers[0].q_w->ne[1] / n_head;

    // zero-filled per layer k/v caches, so masked-out rows can't hold garbage
    std::vector<ggml_tensor *> k_cache(n_layer), v_cache(n_layer);
    for (int il = 0; il < n_layer; il++) {
        k_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0f);
        v_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0f);
    }

    ggml_tensor * out_code_cache = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_codes);
    out_code_cache = cache_set(out_code_cache, 0, code0);

    ggml_tensor * inp_rand0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(inp_rand0, "inp_rand_0");
    ggml_set_input(inp_rand0);

    // TODO: sampling params should come from mtmd_gen_inp, hardcoded for now
    const int   top_k = 50;
    const float top_p = 1.0f;

    prefill(k_cache, v_cache, out_code_cache, h_state, code0_embd, inp_rand0, top_k, top_p);

    for (int g = 1; g < n_acoustic; g++) {
        ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_set_name(inp_rand, ("inp_rand_" + std::to_string(g)).c_str());
        ggml_set_input(inp_rand);
        out_code_cache = step(k_cache, v_cache, out_code_cache, inp_rand, g, top_k, top_p);
    }

    // output 1: the 16 sampled codes. Not read by any caller yet.
    ggml_set_name(out_code_cache, "out_codes");
    ggml_set_output(out_code_cache);
    ggml_build_forward_expand(gf, out_code_cache);

    // output 2 (last node, read by clip_encode()): the sum of all 16
    // codebook embeddings, fed back to the talker backbone for the next frame
    ggml_tensor * out_embd = code0_embd;
    for (int g = 1; g <= n_acoustic; g++) {
        ggml_tensor * code_g = ggml_view_1d(ctx0, out_code_cache, 1, (size_t) g * out_code_cache->nb[1]);

        ggml_tensor * embd_g = ggml_view_2d(ctx0, model.gen_code_embd_w, model.gen_code_embd_w->ne[0], model.gen_code_embd_w->ne[1],
                                            model.gen_code_embd_w->nb[1], (size_t) (g - 1) * model.gen_code_embd_w->nb[2]);
        ggml_tensor * e = ggml_get_rows(ctx0, embd_g, code_g);
        e = ggml_reshape_1d(ctx0, e, e->ne[0]);

        out_embd = ggml_add(ctx0, out_embd, e);
    }
    out_embd = ggml_reshape_2d(ctx0, out_embd, out_embd->ne[0], 1);
    cb(out_embd, "gen_audio_out", -1);

    ggml_build_forward_expand(gf, out_embd);
    return gf;
}
