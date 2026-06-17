# llama.cpp INI Presets

## Introduction

The INI preset feature, introduced in [PR#17859](https://github.com/ggml-org/llama.cpp/pull/17859), allows users to create reusable and shareable parameter configurations for llama.cpp.

### Using Presets with the Server

When running multiple models on the server (router mode), INI preset files can be used to configure model-specific parameters. Please refer to the [server documentation](../tools/server/README.md) for more details.

### Named presets

If you want to define multiple preset configurations for one or more GGUF models, you can create a blank HF repo containing a single `preset.ini` file that references the actual model(s):

```ini
[*]
mmap = 1

[gpt-oss-20b-hf]
hf          = ggml-org/gpt-oss-20b-GGUF
batch-size  = 2048
ubatch-size = 2048
top-p       = 1.0
top-k       = 0
min-p       = 0.01
temp        = 1.0
chat-template-kwargs = {"reasoning_effort": "high"}

[gpt-oss-120b-hf]
hf          = ggml-org/gpt-oss-120b-GGUF
batch-size  = 2048
ubatch-size = 2048
top-p       = 1.0
top-k       = 0
min-p       = 0.01
temp        = 1.0
chat-template-kwargs = {"reasoning_effort": "high"}
```

You can then use it via `llama-cli` or `llama-server`, example:

```sh
llama-server -hf user/repo:gpt-oss-120b-hf
```

Please make sure to provide the correct `hf-repo` for each child preset. Otherwise, you may get error: `The specified tag is not a valid quantization scheme.`
