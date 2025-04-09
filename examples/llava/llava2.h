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

enum llava2_input_chunk_type {
    LLAVA2_INPUT_CHUNK_TYPE_TEXT,
    LLAVA2_INPUT_CHUNK_TYPE_IMAGE,
};

struct llava2_context;
struct llava2_image_tokens_data; // internal data

using llava2_context_ptr           = std::shared_ptr<struct llava2_context>;
using llava2_image_tokens_data_ptr = std::shared_ptr<struct llava2_image_tokens_data>;

// represents raw image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3
struct llava2_bitmap {
    uint32_t nx;
    uint32_t ny;
    std::vector<unsigned char> data;
};

// represents the processed image as tokens (to be encoded)
struct llava2_image_tokens {
    uint32_t nx; // number of tokens in x direction
    uint32_t ny; // number of tokens in y direction
    uint32_t n_tokens; // == nx * ny
    llava2_image_tokens_data_ptr data; // internal data
};

struct llava2_input_chunk {
    llava2_input_chunk_type type;
    std::vector<int32_t> tokens_text;
    llava2_image_tokens tokens_image;
};

struct llava2_context_params {
    bool use_gpu = true;
    int n_threads = 4;
    enum ggml_log_level verbosity = GGML_LOG_LEVEL_INFO;
};

LLAVA2_API llava2_context_ptr llava2_init_from_file(const char * mmproj_fname,
                                                const llama_model * text_model,
                                                const llava2_context_params ctx_params);

// helper function to load an image from a file
LLAVA2_API int32_t llava2_bitmap_init_from_file(const char * fname, llava2_bitmap & output);

// tokenize an input text prompt and an image
// the prompt must have the input image marker <image> in it
// the marker will be replaced with the image tokens
// for example:
//   "here is an image: <image>\ndescribe it in detail."
//   this will gives 3 chunks:
//   1. "here is an image: <start_of_image>"
//   2. <image> (image tokens)
//   3. "<end_of_image>\ndescribe it in detail."
// number of bitmaps must be equal to the number of <image> markers in the prompt
LLAVA2_API int32_t llava2_tokenize(llava2_context_ptr & ctx,
                                std::vector<llava2_input_chunk> & output,
                                const std::string & prompt,
                                bool add_special,
                                bool parse_special,
                                const std::vector<llava2_bitmap> & bitmaps);

LLAVA2_API int32_t llava2_encode(llava2_context_ptr & ctx,
                            const llava2_image_tokens & image_tokens);

LLAVA2_API float * llava2_get_output_embd(llava2_context_ptr & ctx);

#else

static_assert(false && "C header is not yet supported by this library");

#endif

#endif
