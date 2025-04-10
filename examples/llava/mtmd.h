#ifndef LLAVA2_H
#define LLAVA2_H

#include "ggml.h"
#include "llama.h"
#include "clip.h"

#include <vector>
#include <cinttypes>
#include <memory>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAVA2_API __declspec(dllexport)
#        else
#            define LLAVA2_API __declspec(dllimport)
#        endif
#    else
#        define LLAVA2_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAVA2_API
#endif

#ifdef __cplusplus

enum mtmd_input_chunk_type {
    LLAVA2_INPUT_CHUNK_TYPE_TEXT,
    LLAVA2_INPUT_CHUNK_TYPE_IMAGE,
};

struct mtmd_context;
struct mtmd_image_tokens_data; // internal data

using mtmd_context_ptr           = std::shared_ptr<struct mtmd_context>;
using mtmd_image_tokens_data_ptr = std::shared_ptr<struct mtmd_image_tokens_data>;

// represents raw image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3
struct mtmd_bitmap {
    uint32_t nx;
    uint32_t ny;
    std::vector<unsigned char> data;
};

// represents the processed image as tokens (to be encoded)
struct mtmd_image_tokens {
    uint32_t nx; // number of tokens in x direction
    uint32_t ny; // number of tokens in y direction
    uint32_t n_tokens; // == nx * ny
    mtmd_image_tokens_data_ptr data; // internal data
};

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<int32_t> tokens_text;
    mtmd_image_tokens tokens_image;
};

struct mtmd_context_params {
    bool use_gpu = true;
    bool print_timings = true;
    int n_threads = 4;
    enum ggml_log_level verbosity = GGML_LOG_LEVEL_INFO;
    const char * image_marker = "<__image__>";
};

struct mtmd_input_text {
    std::string text;
    bool add_special;
    bool parse_special;
};

// initialize the mtmd context
// return nullptr on failure
LLAVA2_API mtmd_context_ptr mtmd_init_from_file(const char * mmproj_fname,
                                                const llama_model * text_model,
                                                const mtmd_context_params ctx_params);

// helper function to load an image from a file
// returns 0 on success
// this function is thread-safe
LLAVA2_API int32_t mtmd_bitmap_init_from_file(const char * fname, mtmd_bitmap & output);

// tokenize an input text prompt and an image
// the prompt must have the input image marker (default: "<__image__>") in it
// the marker will be replaced with the image tokens
// for example:
//   "here is an image: <__image__>\ndescribe it in detail."
//   this will gives 3 chunks:
//   1. "here is an image: <start_of_image>"
//   2. (image tokens)
//   3. "<end_of_image>\ndescribe it in detail."
// number of bitmaps must be equal to the number of image markers in the prompt
// this function is thread-safe (shared ctx)
LLAVA2_API int32_t mtmd_tokenize(mtmd_context_ptr & ctx,
                                std::vector<mtmd_input_chunk> & output,
                                const mtmd_input_text & text,
                                const std::vector<mtmd_bitmap> & bitmaps);

// returns 0 on success
LLAVA2_API int32_t mtmd_encode(mtmd_context_ptr & ctx,
                            const mtmd_image_tokens & image_tokens);

// get output embeddings from the last encode pass
LLAVA2_API float * mtmd_get_output_embd(mtmd_context_ptr & ctx);

// simple helper to count the total number of tokens from a list of chunks, useful to keep track of n_past
LLAVA2_API size_t mtmd_helper_get_n_tokens(std::vector<mtmd_input_chunk> & chunks);

// helper function that automatically:
// 1. run llama_decode() on text chunks
// 2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
// if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
LLAVA2_API int32_t mtmd_helper_eval(mtmd_context_ptr & ctx,
                                llama_context * lctx,
                                std::vector<mtmd_input_chunk> & chunks,
                                llama_pos pos0,
                                llama_seq_id seq_id,
                                int32_t n_batch);

#else

static_assert(false && "C header is not yet supported by this library");

#endif

#endif
