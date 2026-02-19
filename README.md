# llama.cpp — HIFI Quantisation Fork

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This is a fork of the [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) project, focused on developing **custom quantisation types** — currently the **HIFI family** of quantisation variants.

The HIFI quantisation types aim to deliver better quality at the same (or similar) model sizes compared to the standard quantisation options. This is an **ongoing, actively developed project** and public contributions are welcome.

## Quick start

To build and use HIFI quantised models, follow the detailed instructions in the **[HIFI Build Guide](HIFI_BUILD_GUIDE.md)**, which covers:

- Cloning and building this fork
- Downloading and converting base models
- Creating imatrix files
- Quantising models with the HIFI types
- Running perplexity tests and benchmarks

## About llama.cpp

The upstream `llama.cpp` project enables LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware — locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen — optimised via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantisation for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

For the full upstream project, see [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

<details>
<summary>Supported models</summary>

Typically finetunes of the base models below are supported as well.

#### Text-only

- [X] LLaMA 🦙
- [x] LLaMA 2 🦙🦙
- [x] LLaMA 3 🦙🦙🦙
- [X] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Jamba](https://huggingface.co/ai21labs)
- [X] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [X] [BERT](https://github.com/ggml-org/llama.cpp/pull/5423)
- [X] [Baichuan 1 & 2](https://huggingface.co/models?search=baichuan-inc/Baichuan)
- [X] [Aquila 1 & 2](https://huggingface.co/models?search=BAAI/Aquila)
- [X] [Starcoder models](https://github.com/ggml-org/llama.cpp/pull/3187)
- [X] [MPT](https://github.com/ggml-org/llama.cpp/pull/3417)
- [X] [Bloom](https://github.com/ggml-org/llama.cpp/pull/3553)
- [x] [Yi models](https://huggingface.co/models?search=01-ai/Yi)
- [X] [StableLM models](https://huggingface.co/stabilityai)
- [x] [Deepseek models](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [GPT-2](https://huggingface.co/gpt2)
- [x] [InternLM2](https://huggingface.co/models?search=internlm2)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Command-R models](https://huggingface.co/models?search=CohereForAI/c4ai-command-r)
- [x] [OLMo](https://allenai.org/olmo)
- [x] [OLMo 2](https://allenai.org/olmo)
- [x] [Granite models](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330)
- [x] [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) + [Pythia](https://github.com/EleutherAI/pythia)
- [x] [Bitnet b1.58 models](https://huggingface.co/1bitLLM)
- [x] [Flan T5](https://huggingface.co/models?search=flan-t5)
- [x] [ChatGLM3-6b](https://huggingface.co/THUDM/chatglm3-6b) + [ChatGLM4-9b](https://huggingface.co/THUDM/glm-4-9b)
- [x] [GLM-4-0414](https://huggingface.co/collections/THUDM/glm-4-0414-67f3cbcb34dd9d252707cb2e)
- [x] [SmolLM](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966)
- [x] [RWKV-6](https://github.com/BlinkDL/RWKV-LM)
- [x] [Hunyuan models](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7)

#### Multimodal

- [x] [LLaVA 1.5 models](https://huggingface.co/collections/liuhaotian/llava-15-653aac15d994e992e2677a7e), [LLaVA 1.6 models](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2)
- [x] [Mini CPM](https://huggingface.co/models?search=MiniCPM)
- [x] [Moondream](https://huggingface.co/vikhyatk/moondream2)
- [x] [Qwen2-VL](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)

</details>

## Supported backends

| Backend | Target devices |
| --- | --- |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [BLAS](docs/build.md#blas-build) | All |
| [SYCL](docs/backend/SYCL.md) | Intel and Nvidia GPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [CANN](docs/build.md#cann) | Ascend NPU |

## Key tools

### [`llama-cli`](tools/cli)

A CLI tool for accessing and experimenting with most of `llama.cpp`'s functionality.

```bash
llama-cli -m model.gguf
```

### [`llama-server`](tools/server)

A lightweight, [OpenAI API](https://github.com/openai/openai-openapi) compatible, HTTP server for serving LLMs.

```bash
llama-server -m model.gguf --port 8080
```

### [`llama-perplexity`](tools/perplexity)

A tool for measuring the [perplexity](tools/perplexity/README.md) of a model over a given text — essential for evaluating quantisation quality.

```bash
llama-perplexity -m model.gguf -f file.txt
```

### [`llama-bench`](tools/llama-bench)

Benchmark the performance of inference for various parameters.

```bash
llama-bench -m model.gguf
```

## Contributing

This is an ongoing project and **public contributions are welcome**. Whether it's new quantisation types, performance improvements, bug fixes, or documentation — all contributions are appreciated.

- Open a PR or issue on this repository
- See [CONTRIBUTING.md](CONTRIBUTING.md) for general guidelines (inherited from upstream)
- Read the [HIFI Build Guide](HIFI_BUILD_GUIDE.md) to get familiar with the project workflow

## Upstream documentation

This fork inherits extensive documentation from the upstream project:

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)

## Dependencies

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [stb-image](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [miniaudio.h](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
