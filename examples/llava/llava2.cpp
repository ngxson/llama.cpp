#include "clip.h"
#include "clip-impl.h"
#include "llava2.h"

#include "llama.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static const char * IMG_MARKER = "<image>";

struct llava2_context {
    struct clip_ctx * ctx_clip;
    const struct llama_model * text_model;
    std::vector<float> image_embd_v; // image embedding vector
    int n_threads;

    llava2_context(const char * mmproj_fname, 
                   const struct llama_model * text_model,
                   const struct llava2_context_params & ctx_params) : n_threads(ctx_params.n_threads) {
        clip_context_params ctx_clip_params;
        ctx_clip_params.use_gpu   = ctx_params.use_gpu;
        ctx_clip_params.verbosity = ctx_params.verbosity;
        ctx_clip = clip_init(mmproj_fname, ctx_clip_params);
        if (!ctx_clip) {
            throw std::runtime_error(string_format("Failed to load CLIP model from %s\n", mmproj_fname));
        }
        this->text_model = text_model;
    }

    ~llava2_context() {
        clip_free(ctx_clip);
    }
};

struct llava2_image_tokens_data {
    clip_image_f32_batch_ptr batch_f32; // preprocessed image patches
};

llava2_context_ptr llava2_init_from_file(const char * mmproj_fname,
        const struct llama_model * text_model,
        const struct llava2_context_params ctx_params) {
    try {
        auto ctx = std::make_shared<llava2_context>(mmproj_fname, text_model, ctx_params);
        return ctx;
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return nullptr;
    }
}

int32_t llava2_bitmap_init_from_file(const char * fname, llava2_bitmap & output) {
    clip_image_u8_ptr img_u8(clip_image_u8_init());
    bool ok = clip_image_load_from_file(fname, img_u8.get());
    if (!ok) {
        LOG_ERR("Unable to load image %s\n", fname);
        return 1;
    }
    unsigned char * data = clip_image_u8_get_data(img_u8.get(), &output.nx, &output.ny);
    output.data.resize(output.nx * output.ny * 3);
    std::memcpy(output.data.data(), data, output.nx * output.ny * 3);
    return 0;
}

// copied from common_tokenize
static std::vector<llama_token> llava2_tokenize_text_internal(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }
    return result;
}

int32_t llava2_tokenize(llava2_context_ptr & ctx,
        std::vector<llava2_input_chunk> & output,
        const std::string & prompt,
        bool add_special,
        bool parse_special,
        const std::vector<llava2_bitmap> & bitmaps) {
    auto vocab = llama_model_get_vocab(ctx->text_model);

    std::vector<std::string> parts = string_split_str(prompt, IMG_MARKER);
    output.clear();
    output.reserve(parts.size());

    size_t i_img = 0;

    for (const auto & part : parts) {
        //printf("tokenizing part: %s\n", part.c_str());
        bool add_bos = &parts.front() == &part;
        auto tokens = llava2_tokenize_text_internal(vocab, part, add_special && add_bos, parse_special);
        if (tokens.empty()) {
            continue;
        }
        output.push_back({
            LLAVA2_INPUT_CHUNK_TYPE_TEXT,
            std::move(tokens),
            {},
        });

        if (&parts.back() != &part) {
            // add image token to middle of 2 parts

            if (i_img >= bitmaps.size()) {
                LOG_ERR("%s: error: not enough images for %d parts\n", __func__, (int)parts.size());
                return 2;
            }

            // shim layer
            clip_image_u8_ptr img_u8(clip_image_u8_init());
            img_u8->nx = bitmaps[i_img].nx;
            img_u8->ny = bitmaps[i_img].ny;
            img_u8->buf.resize(bitmaps[i_img].data.size());
            std::memcpy(img_u8->buf.data(), bitmaps[i_img].data.data(), img_u8->nx * img_u8->ny * 3);

            // preprocess image
            clip_image_f32_batch_ptr batch_f32;
            bool ok = clip_image_preprocess(ctx->ctx_clip, img_u8.get(), batch_f32.get());
            if (!ok) {
                LOG_ERR("Unable to preprocess image\n");
                return 1;
            }

            llava2_image_tokens image_tokens;
            //image_tokens.nx = ...;
            //image_tokens.ny = ...;
            image_tokens.n_tokens = clip_n_patches(ctx->ctx_clip); // TODO @ngxson : use clip_n_patches_by_image
            image_tokens.data = std::unique_ptr<llava2_image_tokens_data>(
                new llava2_image_tokens_data{
                    std::move(batch_f32),
                }
            );

            output.push_back({
                LLAVA2_INPUT_CHUNK_TYPE_IMAGE,
                {},
                std::move(image_tokens),
            });
            i_img++;
        }
    }

    return 0;
}

LLAVA2_API int32_t llava2_encode(llava2_context_ptr & ctx,
                            const llava2_image_tokens & image_tokens) {
    ctx->image_embd_v.reserve(image_tokens.n_tokens * clip_n_mmproj_embd(ctx->ctx_clip));
    return clip_image_batch_encode(
        ctx->ctx_clip,
        ctx->n_threads,
        image_tokens.data->batch_f32.get(),
        ctx->image_embd_v.data());
}

LLAVA2_API float * llava2_get_output_embd(llava2_context_ptr & ctx) {
    return ctx->image_embd_v.data();
}
