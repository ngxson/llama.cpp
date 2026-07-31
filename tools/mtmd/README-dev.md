# libmtmd dev guide

## History

Please refer to [multimodal.md](../../docs/multimodal.md) for a broader context.

In short:
- `libmtmd` started as a wrapper around `libllava` / `clip.cpp`
- Various components that used to be in `clip.cpp` are moved progressively to mtmd. For example, preprocessor is now part of mtmd

## Terminologies

- mtmd: **M**ul**T**i**M**o**D**al
- bitmap: representing a raw input data, for example: RGB image, PCM audio
- tiles / slices: for llava-uhd-style models, the preprocessor breaks a large input into smaller square images called tiles or slices
- chunk: a mtmd_input_chunk represents a preprocessed input that can then be passed through `mtmd_encode()`

## Pipeline

A typical pipeline of the core libmtmd is as follows:
- A bitmap (RGB image or PCM audio) is created
- Bitmap and the text prompt is provided to `mtmd_tokenize()` that breaks the input into chunks
    - The tokenizer function first expands a "lazy" bitmap if it finds one. Typically, this is used by video, so that one media token corresponds to one input bitmap
    - For models that support "fused" temporal frames like Qwen-VL, the tokenizer tries to merge pair of consecutive frames into one batch
    - The preprocessor will then be called, which produces a list of chunks
    - Depending on the model itself, special tokens will be injected to separate image chunks (i.e. llava-uhd-style models)
- Multiple bitmaps may be batched together to form a larger `mtmd_batch()`
- Single image or batch is encoded, via `mtmd_encode()` or `mtmd_batch_encode()`
- Get the output embeddings

## Audio generation support

Audio generation is added to mtmd in PR [#26254](https://github.com/ggml-org/llama.cpp/pull/26254)

Currently, we support the 3-stage pipeline below which should cover most TTS models:
1. (Optional) an audio encoder model that converts reference voice into codes or features
2. A backbone model that accepts text prompt and reference voice as input
3. A feature generator model that takes the hidden state from backbone and generate audio features (usually as audio codes or mel-spectrogram)
4. A model that converts audio features to the final PCM waveform

For example, Qwen3-TTS:
1. Reference voice is encoded using ECAPA-TDNN speaker encoder (`speaker_encoder`)
2. Text prompt and reference voice are processed via a backbone (`talker.model`)
3. A model converts sampled semantic token and hidden state from stage 2 into a list of 15 acoustic codes (`talker.code_predictor`)
4. 16 generated codes are converted into waveform (`code2wav`)

### API design constraint

Due to wide variety of audio generation pipelines, the `mtmd_gen_audio` system is designed to be flexible and reusable by new models.

`mtmd_gen_audio` is split into 2 main API:
- Core API `mtmd.h`: handles main inference. Important: the API surface must be stateless; caller must handle state management and audio frame accumulation.
- Helper API `mtmd-helper.h`: provides a model-agnostic stateful API. Usage example can be found in the `tools/tts` directory.

### Checklist for porting new audio generation models to mtmd

1. Establish a list of reusable and missing components from the current mtmd implementation.
2. Sidecar models (code2wav, bigvgan, etc) must live inside the same GGUF file (but can be in different `clip_context` if necessary)
3. Make sure most of the changes happen inside `mtmd-helper-gen.cpp`. A good PR looks like this:
    - 10-20% changes is to add new backbone (text) model and conversion
    - 60% changes inside `mtmd-helper-gen.cpp`
    - 10% changes inside `libmtmd` and `clip.cpp` systems
    - The rest downstream code (CLI, server) should have no changes at all

## Helper

We provide a set of helper functions via `mtmd_helper` to make using libmtmd easier. The helper provides:
- Image, audio and video file decoding (for example, decode raw JPEG into RGB bitmap)
- Manage `llama_batch` and calls to `llama_decode`
