# llama.cpp INI preset

## Introduction

INI preset is a feature that was added in [PR#17859](https://github.com/ggml-org/llama.cpp/pull/17859). The goal is to allow writing reusable and sharable parameter presets in llama.cpp

### Using preset on server

When using multiple models on server (router mode), INI preset file can be used to configure model-specific parameters. Please refer to [server documentations](../tools/server/README.md) for more.

### Using a remote preset

> [!NOTE]
>
> This feature is currently only supported via the `-hf` option

For GGUF models stored on Hugging Face, you can create a file named `preset.ini` in the root directory of the repository that contains specific configurations for the current model.

Example:

```ini
hf-repo-draft = username/my-draft-model-GGUF
temp = 0.5
top-k = 20
top-p = 0.95
```

For security reason, only certain options are allowed. Please refer to [preset.cpp](../common/preset.cpp) for the list of allowed options.

Example usage:

Provided your repo is `username/my-model-with-preset` having a `preset.ini` with the content above.

```sh
llama-cli -hf username/my-model-with-preset

# equivalent to
llama-cli -hf username/my-model-with-preset \
  --hf-repo-draft username/my-draft-model-GGUF \
  --temp 0.5 \
  --top-k 20 \
  --top-p 0.95
```

You can also optionally override preset args by specifying them in the arguments:

```sh
# forcing temp = 0.1
llama-cli -hf username/my-model-with-preset --temp 0.1
```
