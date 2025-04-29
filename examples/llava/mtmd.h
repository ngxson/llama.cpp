#ifndef MTMD_H
#define MTMD_H

#include "ggml.h"
#include "llama.h"
#include "clip.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <vector>
#include <cinttypes>
#include <memory>
#endif

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define MTMD_API __declspec(dllexport)
#        else
#            define MTMD_API __declspec(dllimport)
#        endif
#    else
#        define MTMD_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define MTMD_API
#endif

#define MTMD_DEFAULT_IMAGE_MARKER "<__image__>"

enum mtmd_input_chunk_type {
    MTMD_INPUT_CHUNK_TYPE_TEXT,
    MTMD_INPUT_CHUNK_TYPE_IMAGE,
};

struct mtmd_context;
struct mtmd_image_tokens;

//
// C API
// this is made to closely resemble the C++ API
//

// forward declaration for C API (the actual struct is defined in C++)
struct mtmd_bitmap;
struct mtmd_input_chunk;

struct mtmd_context_params {
    bool use_gpu;
    bool print_timings;
    int n_threads;
    enum ggml_log_level verbosity;
    const char * image_marker;
};

MTMD_API mtmd_context_params mtmd_context_params_default();

// initialize the mtmd context
// return nullptr on failure
MTMD_API mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
                                            const llama_model * text_model,
                                            const mtmd_context_params ctx_params);

MTMD_API void mtmd_free(mtmd_context * ctx);

// get output embeddings from the last encode pass
MTMD_API float * mtmd_get_output_embd(mtmd_context * ctx);

// whether we need to set non-causal mask before llama_decode
MTMD_API bool mtmd_decode_use_non_causal(mtmd_context * ctx);

// mtmd_bitmap
//
// length of data must be nx * ny * 3
// the data is in RGBRGBRGB... format
// the id is optional (can be nullptr), but useful for KV cache tracking
MTMD_API mtmd_bitmap * mtmd_bitmap_init(
    uint32_t nx,
    uint32_t ny,
    const unsigned char * data,
    const char * id, size_t id_len);
MTMD_API uint32_t              mtmd_bitmap_get_nx  (mtmd_bitmap * bitmap);
MTMD_API uint32_t              mtmd_bitmap_get_ny  (mtmd_bitmap * bitmap);
MTMD_API const unsigned char * mtmd_bitmap_get_data(mtmd_bitmap * bitmap);
MTMD_API const char *          mtmd_bitmap_get_id  (mtmd_bitmap * bitmap);
MTMD_API void                  mtmd_bitmap_free    (mtmd_bitmap * bitmap);

// mtmd_input_chunk
//
// the instance can be constructed via mtmd_tokenize()
MTMD_API enum mtmd_input_chunk_type mtmd_input_chunk_get_type        (const mtmd_input_chunk * chunk);
MTMD_API const llama_token *        mtmd_input_chunk_get_tokens_text (const mtmd_input_chunk * chunk, size_t * n_tokens_output);
MTMD_API const mtmd_image_tokens *  mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk);
MTMD_API void                       mtmd_input_chunk_free            (mtmd_input_chunk * chunk);


//
// C++ API
//

#ifdef __cplusplus

struct mtmd_context_deleter {
    void operator()(mtmd_context * val) { mtmd_free(val); }
};
using mtmd_context_ptr = std::unique_ptr<mtmd_context, mtmd_context_deleter>;

// represents raw image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3
struct mtmd_bitmap {
    uint32_t nx;
    uint32_t ny;
    std::vector<unsigned char> data;
    std::string id; // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
};

struct mtmd_image_tokens_deleter {
    void operator()(mtmd_image_tokens * val); // forward declaration
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens, mtmd_image_tokens_deleter>;

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    mtmd_image_tokens_ptr tokens_image;
};

struct mtmd_input_text {
    std::string text;
    bool add_special;
    bool parse_special;
};

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
// return values:
//   0 on success
//   1 on number of images not matching the number of markers
//   2 on image preprocessing error
MTMD_API int32_t mtmd_tokenize(mtmd_context * ctx,
                                std::vector<mtmd_input_chunk> & output,
                                const mtmd_input_text & text,
                                const std::vector<mtmd_bitmap> & bitmaps);

// access mtmd_image_tokens
MTMD_API size_t      mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens);
MTMD_API size_t      mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens);
MTMD_API size_t      mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens);
MTMD_API std::string mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens);
MTMD_API void        mtmd_image_tokens_free(mtmd_image_tokens * image_tokens);

// returns 0 on success
MTMD_API int32_t mtmd_encode(mtmd_context * ctx,
                            const mtmd_image_tokens * image_tokens);



//
// helper functions (can be implemented based on other functions)
//

// helper to count the total number of tokens from a list of chunks, useful to keep track of n_past
MTMD_API size_t mtmd_helper_get_n_tokens(std::vector<mtmd_input_chunk> & chunks);

// helper function that automatically:
// 1. run llama_decode() on text chunks
// 2. run mtmd_encode() on image chunks, then mtmd_get_output_embd() and then llama_decode()
// if any of the mtmd_encode() or llama_decode() calls return non-zero, stop and forward the error
// otherwise, returns 0 on success
MTMD_API int32_t mtmd_helper_eval(mtmd_context * ctx,
                                llama_context * lctx,
                                std::vector<mtmd_input_chunk> & chunks,
                                llama_pos pos0,
                                llama_seq_id seq_id,
                                int32_t n_batch);

// helper function to construct a mtmd_bitmap from a file
// returns 0 on success
// this function is thread-safe
MTMD_API int32_t mtmd_helper_bitmap_init_from_file(const char * fname, mtmd_bitmap & output);

// helper function to construct a mtmd_bitmap from a buffer
// the buffer must be an image in format supported by stb_image (jpg, png, bmp, gif, etc.)
// returns 0 on success
// this function is thread-safe
MTMD_API int32_t mtmd_helper_bitmap_init_from_buf(const unsigned char * buf, size_t len, mtmd_bitmap & output);

#endif


#endif
