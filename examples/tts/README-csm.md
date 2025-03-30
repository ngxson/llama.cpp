# Sesame CSM

To get the GGUF:

```sh
python examples/tts/convert_csm_to_gguf.py

# default output files:
# sesame-csm-backbone.gguf
# sesame-csm-decoder.gguf

# optionally, quantize it
# (lowest scheme is q8_0, it does not make sense to quantize further, quality degrades too much)
python examples/tts/convert_csm_to_gguf.py --outtype q8_0
```

Compile the example:

```sh
cmake --build build -j --target llama-tts-csm
```

Run the example:

```sh
./build/bin/llama-tts-csm -m sesame-csm-backbone.gguf -p "[0]Hello world."
# sesame-csm-backbone.gguf will automatically be loaded
# make sure the place these 2 GGUF files in the same directory

# output file: output.wav
```
