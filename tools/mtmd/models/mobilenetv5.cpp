#include "models.h"

ggml_cgraph * clip_graph_mobilenetv5::build() {

    fprintf(stderr, "\n--- START build_mobilenetv5 ---\n");

    ggml_tensor * inp = build_inp_raw();

    // 1. Stem - Conv2dSame(3, 64, kernel_size=(3, 3), stride=(2, 2))
    ggml_tensor * cur = pad_same_2d(inp, 3, 3, 2, 2);  // Apply SAME padding

    cur = ggml_conv_2d_direct(ctx0, model.mobilenet_stem_conv_w, cur, 2, 2, 0, 0, 1, 1);  // padding=0
    if (model.mobilenet_stem_conv_b) {
        // Bias is [C, 1, 1, 1], need to reshape to [1, 1, C, 1] for broadcasting to [W, H, C, B]
        ggml_tensor * bias = ggml_reshape_4d(ctx0, model.mobilenet_stem_conv_b, 1, 1, cur->ne[2], 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    if (model.mobilenet_stem_norm_w) cur = rms_norm_2d(cur, model.mobilenet_stem_norm_w);
    cur = ggml_gelu(ctx0, cur);


    // 2. Blocks
    std::vector<ggml_tensor*> intermediate_features;
    const int total_blocks = model.mobilenet_blocks.size();
    
    auto is_stage_start = [&](int i) {
        if (i == 0) return true;
        for (int end_idx : model.mobilenet_stage_ends) {
            if (i == end_idx + 1) return true;
        }
        return false;
    };

    auto is_fusion_point = [&](int i) {
        if (model.mobilenet_stage_ends.size() >= 4) {
                if (i == model.mobilenet_stage_ends[2]) return true; // End of Stage 2
                if (i == model.mobilenet_stage_ends[3]) return true; // End of Stage 3
        } else {
            if (i == total_blocks - 1) return true;
        }
        return false;
    };

    for (int i = 0; i < total_blocks; i++) {
        const auto & block = model.mobilenet_blocks[i];
        int stride = is_stage_start(i) ? 2 : 1;

        if (block.s0_conv_exp_w)      cur = build_edge_residual(cur, block, stride);
        else if (block.attn_q_w)      cur = build_mobilenet_attn(cur, block);
        else                          cur = build_inverted_residual(cur, block, stride);

        if (is_fusion_point(i)) {

            intermediate_features.push_back(cur);
        }
    }

    // 3. Multi-Scale Fusion Adapter (MSFA)
    if (!intermediate_features.empty()) {
        
        // A. Reference Resolution: PyTorch implementation uses inputs[0]
        // We assume intermediate_features[0] is the "High Resolution" target.
        // In MobileNet designs, this is typically the feature map with the smallest stride (e.g. 32x32).
        ggml_tensor* target_feat = intermediate_features[0];
        int high_res_w = target_feat->ne[0];
        int high_res_h = target_feat->ne[1];

        std::vector<ggml_tensor*> resized_feats;

        // B. Resize inputs to match inputs[0] (High Resolution)
        for (auto feat : intermediate_features) {
            int feat_w = feat->ne[0];
            int feat_h = feat->ne[1];

            // PyTorch: if feat_size < high_resolution: interpolate
            if (feat_w < high_res_w || feat_h < high_res_h) {
                // Calculate scale factor. 
                // Note: PyTorch 'nearest' works on arbitrary float scales. 
                // ggml_upscale generally takes integer factors or target sizes depending on helper.
                // Assuming standard power-of-2 scaling (e.g. 16 -> 32 means scale=2).
                int scale_w = high_res_w / feat_w;
                // int scale_h = high_res_h / feat_h;
                
                // Safety check for non-integer scaling if strictly replicating
                if (high_res_w % feat_w != 0) { 
                    fprintf(stderr, "Warning: Non-integer scaling detected in MSFA\n"); 
                }

                // Upsample (Nearest Neighbor)
                // 2 is the scale factor
                feat = ggml_upscale(ctx0, feat, scale_w, ggml_scale_mode::GGML_SCALE_MODE_NEAREST); 
            }
            resized_feats.push_back(feat);
        }

        // C. Concatenate at High Resolution (Channel Dim = 2 in ggml)
        cur = resized_feats[0];
        for (size_t k = 1; k < resized_feats.size(); ++k) {
            cur = ggml_concat(ctx0, cur, resized_feats[k], 2);
        }

        // D. FFN (UniversalInvertedResidual)
        // Structure: Expand Conv -> Norm -> GELU -> Project Conv -> Norm
        
        // 1. Expansion
        if (model.msfa_ffn_expand_w) {
            // 1x1 Conv
            cur = ggml_conv_2d_direct(ctx0, model.msfa_ffn_expand_w, cur, 1, 1, 0, 0, 1, 1);
            
            if (model.msfa_ffn_expand_bn) {
                cur = rms_norm_2d(cur, model.msfa_ffn_expand_bn);
            }
            
            cur = ggml_gelu(ctx0, cur);

        }

        // 2. Projection (No DW because kernel_size=0)
        if (model.msfa_ffn_project_w) {
            // 1x1 Conv
            cur = ggml_conv_2d_direct(ctx0, model.msfa_ffn_project_w, cur, 1, 1, 0, 0, 1, 1);
            
            // UniversalInvertedResidual typically has a norm after projection
            if (model.msfa_ffn_project_bn) {
                cur = rms_norm_2d(cur, model.msfa_ffn_project_bn);
            }

        }

        // E. Final Downsample to Target Resolution (Output Resolution)
        // PyTorch: matches self.output_resolution (e.g. 16x16)
        const int target_out_res = 16; 
        int current_w = cur->ne[0];

        if (current_w > target_out_res) {
            int s = current_w / target_out_res;

            if (current_w % target_out_res == 0) {
                // Avg Pool: Kernel=s, Stride=s
                cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, s, s, s, s, 0, 0);
            } else {
                fprintf(stderr, "Error: Irregular downsampling stride required.\n");
            }

        }

        // F. Final Norm
        if (model.msfa_concat_norm_w) {
            cur = rms_norm_2d(cur, model.msfa_concat_norm_w);

        }
    }

    // 4. Gemma 3n Multimodal Projection (Embedder)
    // Input: 'cur' is [Width, Height, Channels, Batch]
    int W = cur->ne[0];
    int H = cur->ne[1];
    int C = cur->ne[2]; // Should be 2048
    int B = cur->ne[3];

    // 1. Permute and Flatten to [Channels, Tokens, Batch]
    // PyTorch expects (Batch, Seq, Hidden), GGML usually processes (Hidden, Seq, Batch)
    cur = ggml_permute(ctx0, cur, 2, 1, 0, 3); // -> [C, H, W, B]
    cur = ggml_permute(ctx0, cur, 0, 2, 1, 3); // -> [C, W, H, B]
    cur = ggml_cont(ctx0, cur);
    cur = ggml_reshape_3d(ctx0, cur, C, W*H, B);
    cur = ggml_cont(ctx0, cur);


    // 2. FEATURE SCALING
    // PyTorch: vision_outputs *= self.config.vision_config.hidden_size**0.5
    // This prevents the signal from vanishing during the subsequent RMSNorm.
    const float scale_factor = sqrtf((float)C);
    cur = ggml_scale(ctx0, cur, scale_factor);


    // 3. SOFT EMBEDDING NORM
    // PyTorch: self._norm(x) * self.weight
    // We must normalize regardless, then multiply if weight exists.
    {
        const float eps = 1e-6f; // Gemma3n uses 1e-6
        cur = ggml_rms_norm(ctx0, cur, eps); 
        
        if (model.mm_soft_emb_norm_w) {
            // Weight shape is (2048,) -> Element-wise broadcast multiply
            cur = ggml_mul(ctx0, cur, model.mm_soft_emb_norm_w);
        }

    }

    // 4. PROJECTION
    // PyTorch: embedding_projection = nn.Linear(vision_hidden, text_hidden, bias=False)
    // Weight stored as [out_features, in_features] = [text_hidden_size, vision_hidden_size]
    if (model.mm_input_proj_w) {
        cur = ggml_mul_mat(ctx0, model.mm_input_proj_w, cur);         
    }

    // 5. POST PROJECTION NORM
    // PyTorch: embedding_post_projection_norm = Gemma3nRMSNorm(..., with_scale=False)
    // with_scale=False means weight is registered as buffer with value 1.0
    // So output = rms_norm(x) * 1.0 = rms_norm(x), magnitude ~1
    {
        const float eps = 1e-6f;
        cur = ggml_rms_norm(ctx0, cur, eps);

        if (model.mm_post_proj_norm_w) {
            // If weight is loaded, multiply (should be ~1.0 anyway)
            cur = ggml_mul(ctx0, cur, model.mm_post_proj_norm_w);
        }
    }

    ggml_build_forward_expand(gf, cur);
    return gf;
}