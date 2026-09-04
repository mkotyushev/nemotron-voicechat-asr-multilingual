"""One naming for three encoders that are the same architecture.

The perception encoder exists here under two sets of names:

  * the original VoiceChat safetensors (and derived GGUFs), under NeMo names
    (`encoder.pre_encode.conv.0`, `self_attn.linear_q`, `conv.batch_norm`);
  * `nvidia/nemotron-*-asr-streaming-*` safetensors, under the HF port's names
    (`encoder.subsampling.conv_in`, `self_attn.q_proj`, `conv.norm`), F32.

`convert_asr_to_mmproj.py` and the fork's stage 2 each map one of them onto the
mmproj's third naming. This module maps both onto the HF one, because that is
the one a reader can check against a published config, and returns plain
`torch.Tensor` state dicts that :mod:`asr_align.encoder` loads directly.

The container is the *only* source of three tensors: `proj`, the 1024 -> 4480
linear into the STT LLM, and the mel filterbank and STFT window, which the HF
repos build in their processor rather than shipping. That asymmetry is the whole
problem this package exists to solve, so the loaders keep it visible:
`load_container` returns them and `load_asr` does not.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

CONTAINER_PREFIX = "stt_model.perception."

# (index in NeMo's flat pre_encode Sequential, HF module path). The gaps are
# ReLU; the HF port splits the same five kernels into a stem and two
# depthwise/pointwise pairs.
SUBSAMPLING_CONVS = (
    (0, "encoder.subsampling.conv_in"),
    (2, "encoder.subsampling.layers.0.depthwise_conv"),
    (3, "encoder.subsampling.layers.0.pointwise_conv"),
    (5, "encoder.subsampling.layers.1.depthwise_conv"),
    (6, "encoder.subsampling.layers.1.pointwise_conv"),
)

# (NeMo name, HF name) inside one encoder block. Everything not listed keeps the
# same name in both.
ATTENTION = (
    ("self_attn.linear_q.weight", "self_attn.q_proj.weight"),
    ("self_attn.linear_k.weight", "self_attn.k_proj.weight"),
    ("self_attn.linear_v.weight", "self_attn.v_proj.weight"),
    ("self_attn.linear_out.weight", "self_attn.o_proj.weight"),
    ("self_attn.linear_pos.weight", "self_attn.relative_k_proj.weight"),
    ("self_attn.pos_bias_u", "self_attn.bias_u"),
    ("self_attn.pos_bias_v", "self_attn.bias_v"),
)

SHARED = (
    "norm_self_att.weight", "norm_self_att.bias",
    "norm_out.weight", "norm_out.bias",
    "norm_feed_forward1.weight", "norm_feed_forward1.bias",
    "norm_feed_forward2.weight", "norm_feed_forward2.bias",
    "norm_conv.weight", "norm_conv.bias",
    "feed_forward1.linear1.weight", "feed_forward1.linear2.weight",
    "feed_forward2.linear1.weight", "feed_forward2.linear2.weight",
    "conv.pointwise_conv1.weight", "conv.pointwise_conv2.weight",
    "conv.depthwise_conv.weight",
)

# NeMo reuses the `batch_norm` attribute name for what conv_norm_type=layer_norm
# makes a LayerNorm; the HF port calls it what it is.
CONV_NORM = (
    ("conv.batch_norm.weight", "conv.norm.weight"),
    ("conv.batch_norm.bias", "conv.norm.bias"),
)

N_MEL = 128

# Tensors both converters write to the mmproj as F16, because ggml_conv_2d and
# ggml_conv_2d_dw_direct want an F16 kernel and the pointwise convolutions are
# matmuls that ride along with them. The source is F32 (safetensors) or Q8_0
# (container) in both cases, so the F16 rounding happens on the way into the
# file and the encoder that actually runs has these weights at half precision.
# Reproducing that is the difference between a port that agrees with the runtime
# to 1e-3 and one that agrees to 1e-4; see check_encoder_parity.py.
# Tensors both converters write at --quant, which is Q8_0 by default. Out of the
# container these arrive already quantized and are copied block for block, so
# rounding them again is a no-op. Out of safetensors they are F32 and the
# converter quantizes them on the way in -- so an encoder read straight from
# safetensors is not the encoder that runs, and a map fitted against it would be
# fitted against a slightly better model than the one being served.
Q8_0_IN_MMPROJ = (
    "encoder.subsampling.linear.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.relative_k_proj.weight",
    "feed_forward1.linear1.weight",
    "feed_forward1.linear2.weight",
    "feed_forward2.linear1.weight",
    "feed_forward2.linear2.weight",
    "proj.weight",
)

F16_IN_MMPROJ = (
    "encoder.subsampling.conv_in.weight",
    "encoder.subsampling.layers.0.depthwise_conv.weight",
    "encoder.subsampling.layers.0.pointwise_conv.weight",
    "encoder.subsampling.layers.1.depthwise_conv.weight",
    "encoder.subsampling.layers.1.pointwise_conv.weight",
    "conv.pointwise_conv1.weight",
    "conv.pointwise_conv2.weight",
)


class EncoderWeights(dict):
    """A canonical state dict plus the hyper-parameters that shape the graph."""

    def __init__(self, tensors: dict[str, torch.Tensor], config: dict, name: str):
        super().__init__(tensors)
        self.config = config
        self.name = name

    @property
    def n_layer(self) -> int:
        return int(self.config["num_hidden_layers"])

    @property
    def n_embd(self) -> int:
        return int(self.config["hidden_size"])

    @property
    def n_head(self) -> int:
        return int(self.config["num_attention_heads"])

    @property
    def conv_kernel_size(self) -> int:
        return int(self.config["conv_kernel_size"])

    @property
    def attention_left_context(self) -> int:
        """clip.cpp's `attn_window_size`: frames to the left, not counting self.

        `sliding_window` counts the frame itself, so this is one less -- 70 for
        VoiceChat and the English model, 56 for the multilingual one. Getting it
        wrong would not crash anything; it would quietly align two encoders that
        are not the ones the deployment runs.
        """

        return int(self.config["sliding_window"]) - 1


def _torch(array: np.ndarray) -> torch.Tensor:
    # copy=True because both readers hand back views onto a read-only mmap or
    # bytes object, and torch warns on every one of those
    return torch.from_numpy(np.array(array, dtype=np.float32, order="C", copy=True))


def q8_0(tensor: torch.Tensor) -> torch.Tensor:
    """ggml's Q8_0, quantized and dequantized again.

    Blocks of 32 share one scale, `amax / 127`, stored as F16 -- but the
    quantization itself divides by the unrounded scale, which is what
    `quantize_row_q8_0` does. On a tensor that was already Q8_0 this is exactly
    the identity: the block maximum is `127 * d`, so the scale comes back as the
    same F16 and every code round-trips to itself. :func:`_assert_idempotent`
    holds it to that.
    """

    # Keep the quantizer arithmetic in F32.  ggml's reference implementation
    # computes both ``d`` and ``1/d`` as floats, and its Python writer mirrors
    # C ``roundf`` (half away from zero).  Torch's default ``round`` is
    # ties-to-even and doing the scale calculation in F64 can move a value
    # across a rounding boundary, so either shortcut can differ by one Q8 code.
    flat = tensor.float().reshape(-1, 32)
    amax = flat.abs().amax(dim=1, keepdim=True)
    unrounded_scale = amax / 127.0
    inverse_scale = torch.where(unrounded_scale != 0, 1.0 / unrounded_scale, 0.0)
    scaled = flat * inverse_scale
    # Spell roundf the same way as gguf-py's ``np_roundf``.  ``floor(x+.5)``
    # is mathematically equivalent but not bit-equivalent in F32: for a value
    # immediately below .5, adding .5 can itself round up to exactly 1.0.
    magnitude = scaled.abs()
    integral = torch.floor(magnitude)
    codes = scaled.sign() * (integral + torch.floor(2.0 * (magnitude - integral)))
    stored_scale = unrounded_scale.to(torch.float16).to(torch.float32)
    return (codes.clamp(-128, 127) * stored_scale).reshape(tensor.shape)


def _as_deployed(name: str, tensor: torch.Tensor, mmproj_precision: bool) -> torch.Tensor:
    """Round a tensor the way writing it to the mmproj would.

    Two rules, and between them they cover everything the converters do not copy
    verbatim: :data:`F16_IN_MMPROJ` because ggml's 2-D convolutions want an F16
    kernel, and :data:`Q8_0_IN_MMPROJ` because that is what --quant says. Norms,
    biases, the depthwise convolution and the two position biases are F32 on both
    sides and pass through.

    Skipping this does not fail loudly. It gives a port that is a slightly more
    accurate encoder than the one being served, which is the worst kind of
    difference: `check_encoder_parity.py` is what turns it into a number.
    """

    if not mmproj_precision:
        return tensor
    if any(name.endswith(suffix) for suffix in F16_IN_MMPROJ):
        return tensor.half().float()
    if tensor.ndim == 2 and any(name.endswith(suffix) for suffix in Q8_0_IN_MMPROJ):
        if tensor.shape[-1] % 32:
            raise ValueError(f"{name}: a row of {tensor.shape[-1]} is not a Q8_0 block")
        return q8_0(tensor)
    return tensor


def load_asr(asr_dir: Path, *, mmproj_precision: bool = True) -> EncoderWeights:
    """The encoder half of a published streaming ASR checkpoint.

    The RNN-T decoder, the joint, `encoder_projector` and (on the multilingual
    model) `prompt_projector` are left behind: none of them is on the path from
    a waveform to what VoiceChat's `proj` reads. `prompt_projector` is the one
    that might deserve to be -- see :func:`load_prompt_projector`.

    A directory written by :mod:`asr_align.export` carries three more things --
    `proj` and the two featurizer tensors -- and those are picked up when they
    are there. That is what makes such a directory self-contained: everything
    the mmproj needs, in one checkpoint, with no container to read.
    """

    st = _safetensors(asr_dir / "model.safetensors")
    config = json.loads((asr_dir / "config.json").read_text())["encoder_config"]

    class _Out(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, _as_deployed(key, value, mmproj_precision))

    out: dict[str, torch.Tensor] = _Out()
    for _, module in SUBSAMPLING_CONVS:
        out[module + ".weight"] = _torch(st.f32(module + ".weight"))
        out[module + ".bias"] = _torch(st.f32(module + ".bias"))
    for name in ("encoder.subsampling.linear.weight", "encoder.subsampling.linear.bias"):
        out[name] = _torch(st.f32(name))

    for il in range(int(config["num_hidden_layers"])):
        p = f"encoder.layers.{il}."
        for _, hf in ATTENTION:
            out[p + hf] = _torch(st.f32(p + hf))
        for name in SHARED:
            out[p + name] = _torch(st.f32(p + name))
        for _, hf in CONV_NORM:
            out[p + hf] = _torch(st.f32(p + hf))

    for name in ("proj.weight", "proj.bias"):
        if name in st:
            out[name] = _torch(st.f32(name))
    for name in ("fb", "window"):
        source = f"preprocessor.featurizer.{name}"
        if source in st:
            value = st.f32(source)
            out["featurizer." + name] = _torch(value.reshape(N_MEL, -1) if name == "fb" else value)

    return EncoderWeights(out, config, asr_dir.name)


def _load_voicechat_perception(take, contains, *, name: str) -> EncoderWeights:
    """Map one NeMo-named VoiceChat tensor source to canonical HF names."""

    n_layer = 0
    while contains(f"{CONTAINER_PREFIX}encoder.layers.{n_layer}.norm_out.weight"):
        n_layer += 1
    if n_layer == 0:
        raise SystemExit(f"{name}: no {CONTAINER_PREFIX}encoder.layers.* tensors")

    out: dict[str, torch.Tensor] = {}
    for index, module in SUBSAMPLING_CONVS:
        out[module + ".weight"] = take(
            f"{CONTAINER_PREFIX}encoder.pre_encode.conv.{index}.weight"
        )
        out[module + ".bias"] = take(
            f"{CONTAINER_PREFIX}encoder.pre_encode.conv.{index}.bias"
        )
    out["encoder.subsampling.linear.weight"] = take(
        f"{CONTAINER_PREFIX}encoder.pre_encode.out.weight"
    )
    out["encoder.subsampling.linear.bias"] = take(
        f"{CONTAINER_PREFIX}encoder.pre_encode.out.bias"
    )

    for il in range(n_layer):
        p = f"encoder.layers.{il}."
        for nemo, hf in ATTENTION:
            out[p + hf] = take(CONTAINER_PREFIX + p + nemo)
        for suffix in SHARED:
            out[p + suffix] = take(CONTAINER_PREFIX + p + suffix)
        for nemo, hf in CONV_NORM:
            out[p + hf] = take(CONTAINER_PREFIX + p + nemo)

    # VoiceChat-only, and the reason this file has two naming adapters.
    out["proj.weight"] = take(f"{CONTAINER_PREFIX}proj.weight")
    out["proj.bias"] = take(f"{CONTAINER_PREFIX}proj.bias")
    # {1, 128, 257} in numpy order; the graph reads it flat as [mel][fft_bin].
    out["featurizer.fb"] = take(
        f"{CONTAINER_PREFIX}preprocessor.featurizer.fb"
    ).reshape(N_MEL, -1)
    out["featurizer.window"] = take(
        f"{CONTAINER_PREFIX}preprocessor.featurizer.window"
    )

    # VoiceChat's own att_context_size is [70, 0]. The graph takes it from the
    # mmproj rather than from a config, so state it the way the two published
    # configs do -- one greater than the left context.
    config = {
        "num_hidden_layers": n_layer,
        "hidden_size": int(out["encoder.subsampling.linear.weight"].shape[0]),
        "num_attention_heads": int(out["encoder.layers.0.self_attn.bias_u"].shape[0]),
        "intermediate_size": int(out["encoder.layers.0.feed_forward1.linear1.weight"].shape[0]),
        "num_mel_bins": N_MEL,
        "subsampling_factor": 8,
        "conv_kernel_size": int(out["encoder.layers.0.conv.depthwise_conv.weight"].shape[-1]),
        "sliding_window": 71,
    }
    return EncoderWeights(out, config, name)


def load_voicechat_safetensors(checkpoint: Path) -> EncoderWeights:
    """Load the original, unquantized VoiceChat perception checkpoint.

    ``checkpoint`` may be the NVIDIA checkpoint directory or its
    ``model.safetensors`` file.  No deployment rounding is simulated here:
    these tensors are the F32 arithmetic/evaluation source.  Quantization is a
    later export step and its result is loaded separately with
    :func:`load_mmproj`.
    """

    path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    st = _safetensors(path)
    return _load_voicechat_perception(
        lambda tensor_name: _torch(st.f32(tensor_name)),
        lambda tensor_name: tensor_name in st,
        name=path.parent.name,
    )


def load_container(container: Path, work: Path, *, mmproj_precision: bool = True) -> EncoderWeights:
    """Load a derived VoiceChat GGUF for runtime-parity or legacy analysis.

    A quantized container is never a valid task-arithmetic source.  This loader
    remains for checking deployed artifacts and reproducing historical results;
    :func:`load_voicechat_safetensors` is the full-precision FT_EN source.
    """

    src = _gguf_source(container, work)

    def take(tensor_name: str) -> torch.Tensor:
        return _torch(src.f32(tensor_name))

    weights = _load_voicechat_perception(
        take,
        lambda tensor_name: tensor_name in src.tensors,
        name=container.name,
    )
    if mmproj_precision:
        for tensor_name, value in list(weights.items()):
            weights[tensor_name] = _as_deployed(tensor_name, value, True)
    return weights


def load_mmproj(
    mmproj: Path,
    work: Path,
    *,
    config: dict,
) -> EncoderWeights:
    """Reload a converted VoiceChat perception GGUF under canonical names.

    Unlike :func:`load_asr` with ``mmproj_precision=True``, this reads the
    actual F16/Q8_0 blocks written by :mod:`convert_asr_to_mmproj`.  Comparison
    1 uses both paths and requires exact equality, turning the in-memory
    deployment-rounding model into a checked property of the real artifact.
    ``config`` must be the complete PT_ML encoder configuration inherited by
    the candidate; graph settings are never borrowed from FT_EN or inferred
    from tensor shapes.
    """

    src = _gguf_source(mmproj, work)

    def take(name: str) -> torch.Tensor:
        return _torch(src.f32(name))

    out: dict[str, torch.Tensor] = {}
    for index, module in SUBSAMPLING_CONVS:
        out[module + ".weight"] = take(f"a.conv1d.{index}.weight")
        out[module + ".bias"] = take(f"a.conv1d.{index}.bias").reshape(-1)
    out["encoder.subsampling.linear.weight"] = take("a.pre_encode.out.weight")
    out["encoder.subsampling.linear.bias"] = take("a.pre_encode.out.bias")

    attention = (
        ("attn_q.weight", "self_attn.q_proj.weight"),
        ("attn_k.weight", "self_attn.k_proj.weight"),
        ("attn_v.weight", "self_attn.v_proj.weight"),
        ("attn_out.weight", "self_attn.o_proj.weight"),
        ("linear_pos.weight", "self_attn.relative_k_proj.weight"),
    )
    norms = (
        ("ln1", "norm_self_att"),
        ("ln2", "norm_out"),
        ("ffn_norm", "norm_feed_forward1"),
        ("ffn_norm_1", "norm_feed_forward2"),
        ("norm_conv", "norm_conv"),
    )
    feed_forwards = (
        ("ffn_up.weight", "feed_forward1.linear1.weight"),
        ("ffn_down.weight", "feed_forward1.linear2.weight"),
        ("ffn_up_1.weight", "feed_forward2.linear1.weight"),
        ("ffn_down_1.weight", "feed_forward2.linear2.weight"),
    )
    for index in range(int(config["num_hidden_layers"])):
        source = f"a.blk.{index}."
        destination = f"encoder.layers.{index}."
        for mmproj_name, canonical_name in attention:
            out[destination + canonical_name] = take(source + mmproj_name)
        out[destination + "self_attn.bias_u"] = take(source + "pos_bias_u")
        out[destination + "self_attn.bias_v"] = take(source + "pos_bias_v")
        for mmproj_name, canonical_name in norms:
            out[destination + canonical_name + ".weight"] = take(
                source + mmproj_name + ".weight"
            )
            out[destination + canonical_name + ".bias"] = take(
                source + mmproj_name + ".bias"
            )
        for mmproj_name, canonical_name in feed_forwards:
            out[destination + canonical_name] = take(source + mmproj_name)
        out[destination + "conv.pointwise_conv1.weight"] = take(
            source + "conv_pw1.weight"
        ).unsqueeze(-1)
        out[destination + "conv.pointwise_conv2.weight"] = take(
            source + "conv_pw2.weight"
        ).unsqueeze(-1)
        out[destination + "conv.depthwise_conv.weight"] = take(
            source + "conv_dw.weight"
        ).unsqueeze(1)
        out[destination + "conv.norm.weight"] = take(source + "conv_norm.weight")
        out[destination + "conv.norm.bias"] = take(source + "conv_norm.bias")

    out["proj.weight"] = take("mm.a.proj.weight")
    out["proj.bias"] = take("mm.a.proj.bias")
    out["featurizer.fb"] = take("a.mel_filters").reshape(N_MEL, -1)
    out["featurizer.window"] = take("a.window")
    return EncoderWeights(out, copy.deepcopy(config), mmproj.name)


def load_prompt_projector(asr_dir: Path, prompt_id: int) -> dict[str, torch.Tensor] | None:
    """The multilingual model's language head, collapsed for one language.

    It concatenates a 128-wide one-hot to every 1024-wide encoder frame and runs
    the result through 1152 -> 2048 -> ReLU -> 1024. For a fixed language the
    one-hot only ever adds one column of `linear_1` to its bias, so what is left
    is an ordinary two-layer MLP on the encoder output.

    It sits after the encoder, so `proj` never saw it and the `voicechat` graph
    has nowhere to put it. Returning it here is how :mod:`asr_align.interface`
    can measure whether it is worth the C++ change that would.
    """

    st = _safetensors(asr_dir / "model.safetensors")
    if "prompt_projector.linear_1.weight" not in st:
        return None
    w1 = _torch(st.f32("prompt_projector.linear_1.weight"))
    b1 = _torch(st.f32("prompt_projector.linear_1.bias"))
    n_embd = w1.shape[1] - 128
    if not 0 <= prompt_id < 128:
        raise SystemExit(f"prompt id {prompt_id} is outside the 128 the model has")
    return {
        "linear_1.weight": w1[:, :n_embd].contiguous(),
        "linear_1.bias": b1 + w1[:, n_embd + prompt_id],
        "linear_2.weight": _torch(st.f32("prompt_projector.linear_2.weight")),
        "linear_2.bias": _torch(st.f32("prompt_projector.linear_2.bias")),
    }


def _safetensors(path: Path):
    """`convert_asr_to_mmproj.SafeTensors`, so there is one reader, not two."""

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from convert_asr_to_mmproj import SafeTensors  # noqa: E402  (same repo)

    return SafeTensors(path)


def _gguf_source(container: Path, work: Path):
    """The fork's GGUF reader, with the Q8_0 dequantizer `convert.sh` patches in."""

    sys.path.insert(0, str(work / "gguf-py"))
    sys.path.insert(0, str(work / "tools" / "voicechat"))
    try:
        import vc_gguf
    except ImportError as error:
        raise SystemExit(
            f"cannot import vc_gguf from {work}: pass --work, or run "
            "./align_setup.sh once to clone the fork"
        ) from error

    if not hasattr(vc_gguf, "dequant_q8_0"):
        raise SystemExit(
            f"{work} has no Q8_0 dequantizer: run ./align_setup.sh, which applies "
            "patches/q8_0-converters.patch to the checkout"
        )
    return vc_gguf.GGUFSource(container)
