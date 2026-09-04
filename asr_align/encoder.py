"""`clip_graph_voicechat::build()` as a PyTorch module.

This is a port, not an interpretation. Every operation below is the one
`tools/mtmd/models/voicechat.cpp` performs, in the same order, and the
hyper-parameters come from the same places clip.cpp reads them from. That
matters more here than it usually would: the alignment this package fits is a
correction for what the *deployed* encoder does to the *deployed* features, so
an encoder that was merely a faithful FastConformer -- NeMo's, or the HF port's
-- would be the wrong reference. `check_encoder_parity.py` holds it to that,
against `MTMD_DEBUG_EMBEDDINGS` from the real runtime.

Four things make this graph its own projector type rather than `parakeet`:

  * the subsampling convolutions pad 2 left and 1 right on **both** the time and
    the frequency axis, so 128 mel bins become 17 and `subsampling.linear` is
    Linear(256 * 17 = 4352, 1024);
  * the depthwise convolution in each block pads kernel - 1 on the left only;
  * attention sees the current frame and a fixed number to its left and nothing
    to its right (NeMo `chunked_limited` with chunk size 1);
  * `conv_norm_type` is layer_norm, so the module NeMo calls `batch_norm` holds
    no running statistics.

The module tree is named to match :mod:`asr_align.weights`, so a state dict
loads with `strict=True` and a missing tensor is an error rather than a
silently random layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .weights import EncoderWeights

LAYER_NORM_EPS = 1e-5
# NeMo's macaron feed forwards are added at half weight.
FFN_SCALE = 0.5
# `sub_lp` / `sub_rp` in voicechat.cpp: NeMo CausalConv2D(kernel 3, stride 2).
SUB_PAD_LEFT = 2
SUB_PAD_RIGHT = 1


@dataclass(frozen=True)
class Hyper:
    n_layer: int
    n_embd: int
    n_head: int
    n_ff: int
    n_mel: int
    conv_kernel: int
    attention_left_context: int
    subsampling_channels: int = 256

    @property
    def d_head(self) -> int:
        return self.n_embd // self.n_head


class Subsampling(nn.Module):
    """Three stride-2 convolutions and the linear that flattens them.

    NeMo keeps one flat `Sequential` here; the HF port splits it into a dense
    stem and two depthwise/pointwise pairs, with the ReLUs where indices 1 and 4
    used to be. Same five kernels either way.
    """

    def __init__(self, hyper: Hyper):
        super().__init__()
        c = hyper.subsampling_channels
        self.conv_in = nn.Conv2d(1, c, 3, stride=2)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "depthwise_conv": nn.Conv2d(c, c, 3, stride=2, groups=c),
                    "pointwise_conv": nn.Conv2d(c, c, 1),
                }
            )
            for _ in range(2)
        )
        n_freq = hyper.n_mel
        for _ in range(3):
            n_freq = n_freq // 2 + 1
        self.linear = nn.Linear(c * n_freq, hyper.n_embd)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """`(batch, n_mel, n_mel_frames)` -> `(batch, n_frames, n_embd)`."""

        # the graph transposes [time, freq] to [freq, time] and convolves with
        # freq on ne0, so here freq is the width axis and time the height one
        x = mel.unsqueeze(1).transpose(2, 3)
        x = F.relu(self.conv_in(self._pad(x)))
        for layer in self.layers:
            x = layer["depthwise_conv"](self._pad(x))
            x = F.relu(layer["pointwise_conv"](x))

        # [freq, time, chan] -> [freq, chan, time]: the flat feature is
        # channel-major with frequency fastest, which is what `linear` was
        # trained on
        batch, chan, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, time, chan * freq)
        return self.linear(x)

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        # ggml_pad_ext(cur, 2, 1, 2, 1, ...) pads ne0 (freq) and ne1 (time)
        return F.pad(x, (SUB_PAD_LEFT, SUB_PAD_RIGHT, SUB_PAD_LEFT, SUB_PAD_RIGHT))


class RelativeSelfAttention(nn.Module):
    """Transformer-XL relative position attention, as NeMo parameterizes it."""

    def __init__(self, hyper: Hyper):
        super().__init__()
        self.hyper = hyper
        n_embd = hyper.n_embd
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"):
            setattr(self, name, nn.Linear(n_embd, n_embd, bias=False))
        self.bias_u = nn.Parameter(torch.zeros(hyper.n_head, hyper.d_head))
        self.bias_v = nn.Parameter(torch.zeros(hyper.n_head, hyper.d_head))

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        batch, time, _ = x.shape
        n_head, d_head = self.hyper.n_head, self.hyper.d_head

        q = self.q_proj(x).view(batch, time, n_head, d_head)
        k = self.k_proj(x).view(batch, time, n_head, d_head)
        v = self.v_proj(x).view(batch, time, n_head, d_head)
        pos = self.relative_k_proj(pos_emb).view(-1, n_head, d_head)

        content = torch.einsum("bqhd,bkhd->bhqk", q + self.bias_u, k)
        relative = torch.einsum("bqhd,whd->bhqw", q + self.bias_v, pos)

        # ggml does this with a pad, a roll, a reshape and two strided views;
        # all four together say that the score for (query q, key k) is the one
        # computed for relative index (n_time - 1) - (q - k), which is what
        # `shift` holds.
        relative = relative.gather(3, shift.expand(batch, n_head, time, time))

        scores = (content + relative) / math.sqrt(d_head)
        probs = torch.softmax(scores + mask, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", probs, v)
        return self.o_proj(out.reshape(batch, time, n_head * d_head))


class FeedForward(nn.Module):
    def __init__(self, hyper: Hyper):
        super().__init__()
        self.linear1 = nn.Linear(hyper.n_embd, hyper.n_ff, bias=False)
        self.linear2 = nn.Linear(hyper.n_ff, hyper.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.silu(self.linear1(x)))


class ConvolutionModule(nn.Module):
    """Pointwise, GLU, causal depthwise, LayerNorm over channels, SiLU, pointwise."""

    def __init__(self, hyper: Hyper):
        super().__init__()
        n_embd, kernel = hyper.n_embd, hyper.conv_kernel
        self.pointwise_conv1 = nn.Conv1d(n_embd, 2 * n_embd, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(n_embd, n_embd, kernel, groups=n_embd, bias=False)
        self.norm = nn.LayerNorm(n_embd, eps=LAYER_NORM_EPS)
        self.pointwise_conv2 = nn.Conv1d(n_embd, n_embd, 1, bias=False)
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pointwise_conv1(x.transpose(1, 2))
        signal, gate = y.chunk(2, dim=1)
        y = signal * torch.sigmoid(gate)
        # causal: kernel - 1 zeros in front, nothing behind
        y = self.depthwise_conv(F.pad(y, (self.kernel - 1, 0)))
        y = self.norm(y.transpose(1, 2))
        y = F.silu(y)
        return self.pointwise_conv2(y.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    """Macaron FFN, attention, convolution, macaron FFN, output norm."""

    def __init__(self, hyper: Hyper):
        super().__init__()
        n_embd = hyper.n_embd
        self.norm_feed_forward1 = nn.LayerNorm(n_embd, eps=LAYER_NORM_EPS)
        self.feed_forward1 = FeedForward(hyper)
        self.norm_self_att = nn.LayerNorm(n_embd, eps=LAYER_NORM_EPS)
        self.self_attn = RelativeSelfAttention(hyper)
        self.norm_conv = nn.LayerNorm(n_embd, eps=LAYER_NORM_EPS)
        self.conv = ConvolutionModule(hyper)
        self.norm_feed_forward2 = nn.LayerNorm(n_embd, eps=LAYER_NORM_EPS)
        self.feed_forward2 = FeedForward(hyper)
        self.norm_out = nn.LayerNorm(n_embd, eps=LAYER_NORM_EPS)

    def forward(
        self,
        x: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        x = x + FFN_SCALE * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, mask, shift)
        x = x + self.conv(self.norm_conv(x))
        x = x + FFN_SCALE * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Encoder(nn.Module):
    def __init__(self, hyper: Hyper):
        super().__init__()
        self.hyper = hyper
        self.subsampling = Subsampling(hyper)
        self.layers = nn.ModuleList(Block(hyper) for _ in range(hyper.n_layer))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.subsampling(mel)
        time = x.shape[1]
        pos_emb = _position_embedding(time, self.hyper.n_embd, x.device, x.dtype)
        mask = _attention_mask(time, self.hyper.attention_left_context, x.device, x.dtype)
        shift = _relative_shift_index(time, x.device)
        for block in self.layers:
            x = block(x, pos_emb, mask, shift)
        return x


class VoiceChatPerception(nn.Module):
    """The encoder, and -- for the container only -- the projection after it.

    `proj` is what turns a 1024-wide encoder frame into the 4480-wide vector the
    STT language model adds to its token embeddings. The published ASR
    checkpoints have no equivalent (their `encoder_projector` goes to a 640-wide
    RNN-T joint), which is why a swapped-in encoder borrows VoiceChat's and why
    it does not fit.
    """

    def __init__(self, hyper: Hyper, projection_dim: int | None = None):
        super().__init__()
        self.hyper = hyper
        self.encoder = Encoder(hyper)
        self.proj = nn.Linear(hyper.n_embd, projection_dim) if projection_dim else None

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            raise RuntimeError("this encoder was built without a projection")
        return self.proj(hidden)


def build(weights: EncoderWeights) -> VoiceChatPerception:
    """Instantiate and load, leaving the featurizer tensors to the caller."""

    hyper = Hyper(
        n_layer=weights.n_layer,
        n_embd=weights.n_embd,
        n_head=weights.n_head,
        n_ff=int(weights.config["intermediate_size"]),
        n_mel=int(weights.config["num_mel_bins"]),
        conv_kernel=weights.conv_kernel_size,
        attention_left_context=weights.attention_left_context,
        subsampling_channels=int(weights["encoder.subsampling.conv_in.weight"].shape[0]),
    )
    state = {k: v for k, v in weights.items() if not k.startswith("featurizer.")}
    projection_dim = state["proj.weight"].shape[0] if "proj.weight" in state else None
    model = VoiceChatPerception(hyper, projection_dim)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _position_embedding(
    time: int, n_embd: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """`pos_emb` in the graph: sine and cosine interleaved, one row per offset.

    clip.cpp fills `pos_freqs` with `exp(-2k ln(10000) / n_state)` and
    `rel_positions` with `n_time - 1 - t`, so row `t` stands for the relative
    offset `n_time - 1 - t`, counting down from `+ (n_time - 1)`.
    """

    d_half = n_embd // 2
    freqs = torch.exp(
        -torch.arange(d_half, device=device, dtype=torch.float32)
        * 2.0
        * math.log(10000.0)
        / n_embd
    )
    offsets = (time - 1) - torch.arange(2 * time - 1, device=device, dtype=torch.float32)
    theta = offsets[:, None] * freqs[None, :]
    out = torch.empty(2 * time - 1, n_embd, device=device, dtype=torch.float32)
    out[:, 0::2] = torch.sin(theta)
    out[:, 1::2] = torch.cos(theta)
    return out.to(dtype)


def _attention_mask(
    time: int, left: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Causal with a fixed left context: `0 <= q - k <= attn_window_size`."""

    q = torch.arange(time, device=device)
    delta = q[:, None] - q[None, :]
    allowed = (delta >= 0) & (delta <= left)
    # clip.cpp uses -1e30 rather than -inf, and softmax of an all-masked row
    # never happens here because the diagonal is always allowed
    return torch.where(allowed, 0.0, -1e30).to(dtype)


def _relative_shift_index(time: int, device: torch.device) -> torch.Tensor:
    """Which row of `pos_emb` scores the pair (query, key)."""

    q = torch.arange(time, device=device)
    return ((time - 1) - (q[:, None] - q[None, :])).view(1, 1, time, time)
