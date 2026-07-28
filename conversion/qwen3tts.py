from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, TYPE_CHECKING

import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, MmprojModel, TextModel, gguf, logger


# torch activation functions used by Qwen3TTSTalkerResizeMLP (config's hidden_act)
_ACT2FN = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


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

    def __init__(self, dir_model: Path, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        talker_config = dict(hparams["talker_config"])
        talker_config["vocab_size"] = talker_config["text_vocab_size"]
        hparams["text_config"] = talker_config
        super().__init__(dir_model, *args, hparams=hparams, **kwargs)
        self._text_proj_buffer = {}

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
        except FileNotFoundError:
            self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        logger.warning("Qwen3-TTS: only the talker backbone is converted; code_predictor and speaker_encoder are skipped")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("talker.") or name.startswith("talker.code_predictor."):
            return None

        name = name[len("talker."):]
        if name == "codec_head.weight":
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # model.codec_embedding.weight belongs to the mmproj (handled separately)
        if name == "model.codec_embedding.weight":
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
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.TOKEN_EMBD), folded)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("Qwen3TTSForConditionalGeneration")
class Qwen3TTSSpeakerEncoderModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

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
        self.gguf_writer.add_audio_projection_dim(self.n_embd_text)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("speaker_encoder."):
            return None

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "res2net_block.blocks." in name:
            assert bid is not None  # the outer stage index, picked up from the tensor name automatically
            xid = int(name.split("res2net_block.blocks.")[1].split(".")[0])
            suffix = "." + name.rsplit(".", 1)[1]
            new_name = gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.A_ENC_CONV_RES2].format(bid=bid, xid=xid) + suffix
            yield (new_name, data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)
