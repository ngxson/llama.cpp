from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, TextModel, gguf, logger

# Tricks being used to support this model via existing llama.cpp code paths:
# - Text projection MLP is folded into the embedding table
# - codec_embedding is concat to the text embedding table, vocab is extended
#   example: codec_bos_id(2149) --> "<|codec_bos|>"
#            codec_eos_token_id(2150) --> "<|codec_eos_token|>"
#            codec_language_id.chinese(2055) --> "<|codec_language_chinese|>"
#            other rows --> "<|codec_0|>", "<|codec_1|>", ..., "<|codec_1023|>"
# - output tensor codec_head is smaller than vocab, so logits will be padded at inference time
# - suppress_tokens is used to limit the backbone to only sample either semantic or EOS (stop) token

# torch activation functions used by Qwen3TTSTalkerResizeMLP (config's hidden_act)
_ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


# TODO: figure out the correct template
DEFAULT_TEMPLATE = """{% for m in messages %}{{m['content']}}{% endfor %}"""


@ModelBase.register("Qwen3TTSForConditionalGeneration")
class Qwen3TTSTalkerModel(TextModel):
    model_arch = gguf.MODEL_ARCH.QWEN3TTS

    _TEXT_PROJ_KEYS = (
        "model.text_embedding.weight",
        "text_projection.linear_fc1.weight",
        "text_projection.linear_fc1.bias",
        "text_projection.linear_fc2.weight",
        "text_projection.linear_fc2.bias",
    )

    _text_proj_buffer: dict[str, Tensor]
    _folded_text_embed: Tensor | None
    _codec_embed: Tensor | None

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        raw_talker_config = dict(hparams["talker_config"])
        self._talker_config = raw_talker_config
        self.n_codec_vocab = raw_talker_config["vocab_size"]
        talker_config = dict(raw_talker_config)
        talker_config["vocab_size"] = talker_config["text_vocab_size"]
        hparams["text_config"] = talker_config
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self._text_proj_buffer = {}
        self._folded_text_embed = None
        self._codec_embed = None

    def _codec_token_names(self) -> list[str]:
        # start every row with a generic name, then override the ones with a
        # known meaning (bos/eos/language/etc, derived from the *_id fields
        # of talker_config) with a more descriptive one
        names = [f"<|codec_{i}|>" for i in range(self.n_codec_vocab)]
        for key, val in self._talker_config.items():
            if not key.endswith("_id"):
                continue
            prefix = key[:-len("_id")]
            if isinstance(val, int):
                names[val] = f"<|{prefix}|>"
            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    names[subval] = f"<|{prefix}_{subkey}|>"
        return names

    def set_vocab(self):
        codec_tokens = self._codec_token_names()
        codec_toktypes = [gguf.TokenType.CONTROL] * len(codec_tokens)

        try:
            tokens, scores, toktypes = self._create_vocab_sentencepiece()
            self.gguf_writer.add_tokenizer_model("llama")
            self.gguf_writer.add_tokenizer_pre("default")
            tokens += [t.encode("utf-8") for t in codec_tokens]
            scores += [0.0] * len(codec_tokens)
            toktypes += codec_toktypes
            self.gguf_writer.add_token_list(tokens)
            self.gguf_writer.add_token_scores(scores)
            self.gguf_writer.add_token_types(toktypes)
            special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
            special_vocab.add_to_gguf(self.gguf_writer)
            return
        except FileNotFoundError:
            pass

        tokens, toktypes, tokpre = self.get_vocab_base()
        tokens += codec_tokens
        toktypes += codec_toktypes
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_chat_template(DEFAULT_TEMPLATE)

        # note: final vocab layout is [text_vocab | codec_vocab], with text_vocab is actually padded with -inf in cgraph
        # for codec_vocab, only first 2048 rows can be sampled for semantic code
        # plus codec_eos_token_id that used for signaling end of generation
        # ref: https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/qwen_tts/core/models/modeling_qwen3_tts.py#L2059-L2063

        vocab_size = self.hparams["vocab_size"] + self.n_codec_vocab
        codec_eos_token_id = self.hparams["vocab_size"] + self._talker_config["codec_eos_token_id"]
        self.gguf_writer.add_suppress_tokens([
            i for i in range(vocab_size - 1024, vocab_size)
            if i != codec_eos_token_id
        ])
        self.gguf_writer.add_eos_token_id(codec_eos_token_id)
        self.gguf_writer.add_add_eos_token(False)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("talker.") or name.startswith("talker.code_predictor."):
            return None

        name = name[len("talker."):]
        return super().filter_tensors((name, gen))

    def _maybe_emit_token_embd(self) -> Iterable[tuple[str, Tensor]]:
        if self._folded_text_embed is None or self._codec_embed is None:
            return
        combined = torch.cat([self._folded_text_embed, self._codec_embed], dim=0)
        yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), combined)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # codec_embedding rows are appended after the text vocab, extending the embedding table
        if name == "model.codec_embedding.weight":
            self._codec_embed = data_torch
            yield from self._maybe_emit_token_embd()
            return

        # codec_head is the output head for the (smaller) codec vocab; logits get padded to
        # the extended vocab size at inference time
        if name == "codec_head.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch)
            return

        if name in self._TEXT_PROJ_KEYS:
            self._text_proj_buffer[name] = data_torch
            if len(self._text_proj_buffer) < len(self._TEXT_PROJ_KEYS):
                return

            # fold MLP into the embedding table at conversion time, MLP won't be used at inference time anyway
            act_fn = _ACT2FN[self.hparams["hidden_act"]]
            embed = self._text_proj_buffer["model.text_embedding.weight"]
            hidden = act_fn(F.linear(embed,
                                      self._text_proj_buffer["text_projection.linear_fc1.weight"],
                                      self._text_proj_buffer["text_projection.linear_fc1.bias"]))
            folded = F.linear(hidden,
                               self._text_proj_buffer["text_projection.linear_fc2.weight"],
                               self._text_proj_buffer["text_projection.linear_fc2.bias"])
            self._folded_text_embed = folded
            yield from self._maybe_emit_token_embd()
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3TTSForConditionalGeneration")
class Qwen3TTSSpeakerEncoderModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    # talker.code_predictor.model.layers.{bid}.<key> -> A_GEN_CODE_*
    # bypass tensor_mapping.py for now to make it simple
    _CODE_LAYER_TENSOR_MAP = {
        "input_layernorm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_NORM,
        "self_attn.q_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q,
        "self_attn.q_norm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_Q_NORM,
        "self_attn.k_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K,
        "self_attn.k_norm": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_K_NORM,
        "self_attn.v_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_V,
        "self_attn.o_proj": gguf.MODEL_TENSOR.A_GEN_CODE_ATTN_OUT,
        "post_attention_layernorm": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_NORM,
        "mlp.gate_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_GATE,
        "mlp.up_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_UP,
        "mlp.down_proj": gguf.MODEL_TENSOR.A_GEN_CODE_FFN_DOWN,
    }

    # note: codebook pages will be stacked to 3D
    _CODE_GEN_N_CODEBOOKS = 15
    _code_embed_buffer: dict[int, Tensor] = {}
    _code_head_buffer: dict[int, Tensor] = {}

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        hparams["text_config"] = {"hidden_size": hparams["talker_config"]["hidden_size"]}
        # ECAPA-TDNN has a fixed 4-stage backbone, not a configurable transformer depth;
        # MmprojModel.__init__ still needs one of the n_block_keys to build its tensor map
        hparams["speaker_encoder_config"]["n_layers"] = 4
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)

    def get_audio_config(self) -> dict[str, Any] | None:
        return self.global_config.get("speaker_encoder_config")

    def set_gguf_parameters(self):
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_audio_encoder(True)
        self.gguf_writer.add_clip_audio_projector_type(gguf.VisionProjectorType.QWEN3TTS_SPKENC)

        # handle speaker encoder config
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)
        # mel_spectrogram() front-end: sr=24000, n_fft=1024, hop=256, n_mels=128, fmin=0, fmax=12000 (=sr/2, the clip.cpp default)
        self.gguf_writer.add_audio_num_mel_bins(128)
        # the 3 SE-Res2Net stages (blocks 1-3); the stem conv, mfa, asp and fc are singletons, not part of this count
        self.gguf_writer.add_audio_block_count(3)
        # ECAPA-TDNN has no attention/FFN, these are dummy to allow clip.cpp to load it
        self.gguf_writer.add_audio_embedding_length(1536)
        self.gguf_writer.add_audio_head_count(1)
        self.gguf_writer.add_audio_feed_forward_length(1536)
        self.gguf_writer.add_audio_attention_layernorm_eps(1e-5)

        # handle code predictor config
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.QWEN3TTS_GEN)
        code_predictor_config = self.global_config["talker_config"]["code_predictor_config"]
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(code_predictor_config["hidden_size"])
        self.gguf_writer.add_gen_audio_feed_forward_length(code_predictor_config["intermediate_size"])
        self.gguf_writer.add_gen_audio_block_count(code_predictor_config["num_hidden_layers"])
        self.gguf_writer.add_gen_audio_head_count(code_predictor_config["num_attention_heads"])
        self.gguf_writer.add_gen_audio_head_count_kv(code_predictor_config["num_key_value_heads"])
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(code_predictor_config["rms_norm_eps"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not (
            name.startswith("speaker_encoder.")
            or name.startswith("talker.code_predictor.")
            or name == "talker.model.codec_embedding.weight"
        ):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # codebook-0 embedding: fed back into the talker backbone once a codec token is
        # generated, the counterpart of code_predictor's codec_embedding.{0..14} for codebooks 1-15
        if name == "talker.model.codec_embedding.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_OUT_EMBD), data_torch)
            return

        if name == "talker.code_predictor.model.norm.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_OUTPUT_NORM), data_torch)
            return

        if name.startswith("talker.code_predictor.small_to_mtp_projection."):
            suffix = "." + name.rsplit(".", 1)[1]
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_PROJ_IN, suffix=suffix), data_torch)
            return

        if name.startswith("talker.code_predictor.model.codec_embedding."):
            idx = int(name.split("codec_embedding.")[1].split(".")[0])
            self._code_embed_buffer[idx] = data_torch
            if len(self._code_embed_buffer) < self._CODE_GEN_N_CODEBOOKS:
                return
            stacked = torch.stack([self._code_embed_buffer.pop(i) for i in range(self._CODE_GEN_N_CODEBOOKS)], dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_EMBD), stacked)
            return

        if name.startswith("talker.code_predictor.lm_head."):
            idx = int(name.split("lm_head.")[1].split(".")[0])
            self._code_head_buffer[idx] = data_torch
            if len(self._code_head_buffer) < self._CODE_GEN_N_CODEBOOKS:
                return
            stacked = torch.stack([self._code_head_buffer.pop(i) for i in range(self._CODE_GEN_N_CODEBOOKS)], dim=0)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.A_GEN_CODE_HEAD), stacked)
            return

        if name.startswith("talker.code_predictor.model.layers."):
            rest = name.split("model.layers.")[1]        # "{bid}.<key>.weight"
            _, key_with_suffix = rest.split(".", 1)       # "<key>.weight"
            key = key_with_suffix.rsplit(".", 1)[0]        # "<key>"
            tensor = self._CODE_LAYER_TENSOR_MAP.get(key)
            if tensor is not None:
                yield (self.format_tensor_name(tensor, bid), data_torch)
                return

        if "res2net_block.blocks." in name:
            assert bid is not None  # the outer stage index, picked up from the tensor name automatically
            xid = int(name.split("res2net_block.blocks.")[1].split(".")[0])
            suffix = "." + name.rsplit(".", 1)[1]
            new_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.A_ENC_CONV_RES2].format(bid=bid, xid=xid) + suffix
            yield (new_name, data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)
