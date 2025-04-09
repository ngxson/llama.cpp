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

struct llava2_context {
    struct clip_ctx * ctx_clip;
    const struct llama_model * text_model;
    std::vector<float> image_embd_v; // image embedding vector
    int n_threads;
    std::string image_marker;

    // TODO @ngxson : add timings

    llava2_context(const char * mmproj_fname,
                   const struct llama_model * text_model,
                   const struct llava2_context_params & ctx_params) : n_threads(ctx_params.n_threads), image_marker(ctx_params.image_marker) {
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

    std::vector<std::string> parts = string_split_str(prompt, ctx->image_marker);
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
            clip_image_f32_batch_ptr batch_f32(new clip_image_f32_batch);
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
    int n_mmproj_embd = clip_n_mmproj_embd(ctx->ctx_clip);
    ctx->image_embd_v.resize(image_tokens.n_tokens * n_mmproj_embd);
    bool ok = clip_image_batch_encode(
        ctx->ctx_clip,
        ctx->n_threads,
        image_tokens.data->batch_f32.get(),
        ctx->image_embd_v.data());
    return ok ? 0 : 1;
}

LLAVA2_API float * llava2_get_output_embd(llava2_context_ptr & ctx) {
    return ctx->image_embd_v.data();
}

size_t llava2_helper_get_n_tokens(std::vector<llava2_input_chunk> & chunks) {
    size_t n_tokens = 0;
    for (auto & chunk : chunks) {
        if (chunk.type == LLAVA2_INPUT_CHUNK_TYPE_TEXT) {
            n_tokens += chunk.tokens_text.size();
        } else if (chunk.type == LLAVA2_INPUT_CHUNK_TYPE_IMAGE) {
            n_tokens += chunk.tokens_image.n_tokens;
        } else {
            GGML_ASSERT(false && "chunk type not supported");
        }
    }
    return n_tokens;
}

// helper struct to make working with embd batch easier
// note: this will be removed after llama_batch_ext refactoring
struct decode_embd_batch {
    std::vector<llama_pos>      pos;
    std::vector<int32_t>        n_seq_id;
    std::vector<llama_seq_id>   seq_id_0;
    std::vector<llama_seq_id *> seq_ids;
    std::vector<int8_t>         logits;
    llama_batch batch;
    decode_embd_batch(float * embd, int32_t n_tokens, llama_pos pos_0, llama_seq_id seq_id) {
        pos     .resize(n_tokens);
        n_seq_id.resize(n_tokens);
        seq_ids .resize(n_tokens + 1);
        logits  .resize(n_tokens);
        seq_id_0.resize(1);
        seq_id_0[0] = seq_id;
        seq_ids [n_tokens] = nullptr;
        batch = {
            /*n_tokens       =*/ n_tokens,
            /*tokens         =*/ nullptr,
            /*embd           =*/ embd,
            /*pos            =*/ pos.data(),
            /*n_seq_id       =*/ n_seq_id.data(),
            /*seq_id         =*/ seq_ids.data(),
            /*logits         =*/ logits.data(),
        };
        for (int i = 0; i < n_tokens; i++) {
            batch.pos     [i] = pos_0 + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id  [i] = seq_id_0.data();
            batch.logits  [i] = false;
        }
    }
};

int32_t llava2_helper_eval(llava2_context_ptr & ctx,
        llama_context * lctx,
        std::vector<llava2_input_chunk> & chunks,
        llama_pos pos0,
        llama_seq_id seq_id,
        int32_t n_batch) {
    int32_t ret;
    llama_pos n_past = pos0;
    llama_batch text_batch = llama_batch_init(n_batch, 0, 1);

    for (auto & chunk : chunks) {
        bool is_last = &chunk == &chunks.back();
        if (chunk.type == LLAVA2_INPUT_CHUNK_TYPE_TEXT) {
            // TODO @ngxson : may need to split into smaller batches
            text_batch.n_tokens = chunk.tokens_text.size();
            for (size_t i = 0; i < chunk.tokens_text.size(); i++) {
                text_batch.token   [i]    = chunk.tokens_text[i];
                text_batch.pos     [i]    = n_past++;
                text_batch.n_seq_id[i]    = 1;
                text_batch.seq_id  [i][0] = seq_id;
                text_batch.logits  [i]    = false;
            }
            if (is_last) {
                // always get logits for last input chunk
                text_batch.logits[text_batch.n_tokens - 1] = true;
            }
            ret = llama_decode(lctx, text_batch);
            if (ret != 0) {
                LOG_ERR("failed to decode text\n");
                llama_batch_free(text_batch);
                return ret;
            }

        } else if (chunk.type == LLAVA2_INPUT_CHUNK_TYPE_IMAGE) {
            GGML_ASSERT(!is_last && "logits for last image chunk is not yet support");
            ret = llava2_encode(ctx, chunk.tokens_image);
            if (ret != 0) {
                LOG_ERR("failed to encode image\n");
                llama_batch_free(text_batch);
                return ret;
            }

            int32_t n_tokens = chunk.tokens_image.n_tokens;
            float * embd = llava2_get_output_embd(ctx);
            decode_embd_batch batch_img(embd, n_tokens, n_past, 0);
            ret = llama_decode(lctx, batch_img.batch);
            if (ret != 0) {
                LOG_ERR("failed to decode image\n");
                llama_batch_free(text_batch);
                return ret;
            }

            n_past += n_tokens;

        } else {
            GGML_ASSERT(false && "chunk type not supported");
        }
    }

    llama_batch_free(text_batch);
    return 0;
}
