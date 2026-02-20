# Requirements

- transformers: pip install transformers
- torch: pip install torch
- huggingface-cli: curl -LsSf https://hf.co/cli/install.sh | bash 
- sentencepiece: pip install sentencepiece

# How to build a HIFI model

The HIFI family of quantisation variants are available through a custom fork of the llama.cpp project.

You will need to download and build this on your own server or computer:

To download, clone the project:
```bash
git clone https://github.com/geoffmunn/llama.cpp.git
cd llama.cpp
```

## Hardware support requirements

If you only want a CPU version, you can skip these requirements. Otherwise, add anything you might need.

**MacOS**

No extra requirements, Apple Silicon should work if you have Xcode 16 (or 15).

**Windows**

Vulkan support if you think you need it, otherwise a CPU build will work

- nVidia CUDA toolkit
- Vulkan SDK
- Long filenames support enabled in Windows (required if you install the Vulkan SDK)

**Raspberry Pi**

No extra requirements, but it will be slow :)

**nVidia AI server**

No extra requirements but it will depend on your hardware configuration.

## Build steps

### Base image

First, you'll need the base image that you'll be building this off. **REPLACE `0.6B` WITH THE VERSION YOU WANT**

Windows:
```powershell
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
python .\convert_hf_to_gguf.py .\Qwen3-0.6B\ --outfile .\Qwen3-0.6B-f16.gguf --outtype f16
```

Linux & MacOS:
```bash
hf download Qwen/Qwen3-0.6B --local-dir ./Qwen3-0.6B
python3 ./convert_hf_to_gguf.py ./Qwen3-0.6B/ --outfile ./Qwen3-0.6B-f16.gguf --outtype f16
```

### Wikitext

Now download and extract wikitext into `.\wikitext-2-raw`. We need this for perplexity testing.

Windows:
```powershell
New-Item -ItemType Directory -Path "wikitext-2-raw" -Force
Invoke-WebRequest -Uri "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip" -OutFile "wikitext-2-raw\wikitext-2-raw-v1.zip"
Expand-Archive -Path "wikitext-2-raw\wikitext-2-raw-v1.zip" -DestinationPath "wikitext-2-raw" -Force
Remove-Item "wikitext-2-raw\wikitext-2-raw-v1.zip"
```

Linux & MacOS:
```bash
mkdir -p wikitext-2-raw
curl -L -o wikitext-2-raw/wikitext-2-raw-v1.zip "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
unzip -o wikitext-2-raw/wikitext-2-raw-v1.zip -d wikitext-2-raw
rm wikitext-2-raw/wikitext-2-raw-v1.zip
```

### Build the project

A regular build looks like this:

**Windows AND Linux**:
```bash
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release -j
```

**MacOS**:
```bash
mkdir build
cmake -B build -DCMAKE_CXX_STANDARD=17 -DGGML_METAL=ON -DGGML_ACCELERATE=OFF -DGGML_BLAS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you want a pure CPU build, then run this (Linux example):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF
```

### Create an imatrix file

### Download the imatrix source files:

There are two purpose-built scripts in the tools directory to help do this.

By default, it will create an imatrix with 4697 chunks which is very large and slow. You can adjust the ratios to reflect your target usage model.

**Windows**:
```powershell
@TODO
```

**Linux & MacOS**:
```bash
chmod +x ./tools/download_imatrix_datasets.py
chmod +x ./tools/create_mixed_imatrix_dataset.py

python3 ./tools/download_imatrix_datasets.py
python3 ./tools/create_mixed_imatrix_dataset.py --wikitext wikitext.txt --code codeparrot.txt --math mathqa.txt --output mixed-imatrix-dataset.txt --ratio 60,25,15
```

**Note: this will take a long time. Take a copy of this file if you want to use it again.**

**Windows**:
```powershell
.\build\bin\Release\llama-imatrix.exe -m .\Qwen3-0.6B-f16.gguf -f ./mixed-imatrix-dataset.txt -o .\Qwen3-0.6B-f16-imatrix-4697.gguf --output-frequency 20 --chunks 5000
```

**Linux & MacOS**:
```bash
./build/bin/llama-imatrix -m ./Qwen3-0.6B-f16.gguf -f ./mixed-imatrix-dataset.txt -o ./Qwen3-0.6B-f16-imatrix-4697.gguf --output-frequency 20 --chunks 5000
```

If your terminal session is likely to expire, then use this long running command:
```bash
nohup ./build/bin/llama-imatrix -m ./Qwen3-32B-f16.gguf -f ./mixed-imatrix-dataset.txt -o ./Qwen3-32B-f16-imatrix-4697.gguf --output-frequency 20 --chunks 5000 -ngl 0 > output.log 2>&1 &
```

### Create a quantised model

**Windows**:

With an imatrix file:
```powershell
.\build\bin\Release\llama-quantize.exe --imatrix .\Qwen3-0.6B-f16-imatrix-4697.gguf .\Qwen3-0.6B-f16.gguf .\Qwen3-0.6B-f16-Q3_K_HIFI.gguf Q3_K_HIFI
```

And without:
```powershell
.\build\bin\Release\llama-quantize.exe .\Qwen3-0.6B-f16.gguf .\Qwen3-0.6B-f16-Q3_K_HIFI.gguf Q3_K_HIFI
```

**Linux & MacOS**:

With an imatrix file:

```bash
./build/bin/llama-quantize --imatrix ./Qwen3-0.6B-f16-imatrix-4697.gguf ./Qwen3-0.6B-f16.gguf ./Qwen3-0.6B-f16-imatrix:Q3_K_HIFI.gguf Q3_K_HIFI
```

And without:
```bash
./build/bin/llama-quantize ./Qwen3-0.6B-f16.gguf ./Qwen3-0.6B-f16:Q3_K_HIFI.gguf Q3_K_HIFI
```

### Perplexity test

**Windows**:
```powershell
.\build\bin\Release\llama-perplexity.exe -m .\Qwen3-0.6B-f16-Q3_HIFI.gguf -f .\wikitext-2-raw\wikitext-2-raw\wiki.test.raw --ppl-stride 0 -c 512
```

**Linux & MacOS**:

```bash
./build/bin/llama-perplexity -m ./Qwen3-0.6B-f16\:Q3_K_HIFI.gguf -f ./wikitext-2-raw/wikitext-2-raw/wiki.test.raw --ppl-stride 0 -c 512
```

### Benchmarking

A single benchmark can be obtained with this command:

**Windows**:
```powershell
.\build\bin\Release\llama-bench.exe -m .\Qwen3-0.6B-f16-Q3_K_S.gguf,.\Qwen3-0.6B-f16-Q3_K_M.gguf,.\Qwen3-0.6B-f16-Q3_K_HIFI.gguf -t 4 -r 3 -p 0 -n 20
```

**Linux & MacOS**:
```bash
./build/bin/llama-bench -m .\Qwen3-0.6B-f16-Q3_K_S.gguf,.\Qwen3-0.6B-f16-Q3_K_M.gguf,.\Qwen3-0.6B-f16-Q3_K_HIFI.gguf -t 4 -r 3 -p 0 -n 20
```

But an average is more useful to smooth out random variations due to CPU load etc. This will make 100 speed tests across all the models listed inside the script, and give you average result.

**Windows**:
```powershell
.\benchmark_speed_test.ps1
```

**Linux & MacOS**:
```bash
./benchmark_speed_test.sh
```

### Upload to Hugging Face

```bash
hf upload geoffmunn/Qwen3-0.6B-f16 ./Qwen3-0.6B-f16-imatrix-4697.gguf Qwen3-0.6B-f16-imatrix-4697.gguf --repo-type model --commit-message "Upload imatrix gguf"
hf upload geoffmunn/Qwen3-0.6B-f16 ./Qwen3-0.6B-f16:Q5_K_HIFI.gguf Qwen3-0.6B-f16:Q5_K_HIFI.gguf --repo-type model --commit-message "Upload Q5_K_HIFI quantized model"
hf upload geoffmunn/Qwen3-0.6B-f16 ./Qwen3-0.6B-f16-imatrix:Q5_K_HIFI.gguf Qwen3-0.6B-f16-imatrix:Q5_K_HIFI.gguf --repo-type model --commit-message "Upload Q5_K_HIFI + imatrix quantized model"
hf upload geoffmunn/Qwen3-0.6B-f16 ./Qwen3-0.6B-f16-imatrix:Q5_K_M.gguf Qwen3-0.6B-f16-imatrix:Q5_K_M.gguf --repo-type model --commit-message "Upload Q5_K_M + imatrix quantized model"
hf upload geoffmunn/Qwen3-0.6B-f16 ./Qwen3-0.6B-f16-imatrix:Q5_K_S.gguf Qwen3-0.6B-f16-imatrix:Q5_K_S.gguf --repo-type model --commit-message "Upload Q5_K_S + imatrix quantized model"
hf upload geoffmunn/Qwen3-0.6B-f16 ./mixed-imatrix-dataset.txt mixed-imatrix-dataset.txt --repo-type model --commit-message "imatrix dataset"
```
