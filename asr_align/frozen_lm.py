"""Differentiable frozen VoiceChat STT backbone, including its duplex inputs.

The source is the original VoiceChat safetensors, never the Nano base weights.
Nemotron-H's Mamba2, unpositioned causal attention, squared-ReLU MLP and RMS
normalizations follow the pinned NVIDIA configuration and deployment graph.
Prefix states are immutable values: checkpoint recomputation must never append
to a shared cache. This also avoids the upstream inference-only cache updates
which cannot propagate gradients from the response back to audio frames.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .experiments import ExperimentValidationError


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, groups: int = 1) -> torch.Tensor:
    dtype, shape = x.dtype, x.shape
    y = x.float().reshape(*shape[:-1], groups, shape[-1] // groups)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y.reshape(shape) * weight.float()).to(dtype)


def ssd_scan(x: torch.Tensor, dt: torch.Tensor, a: torch.Tensor,
             b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
             initial: torch.Tensor | None = None, *, chunk_size: int = 32):
    """Mamba2 recurrence in bounded chunks, with differentiable initial states.

    x [batch,time,heads,head_dim]; b/c [batch,time,groups,state_dim].
    Grouped B/C heads are contiguous, as in the runtime and Triton kernels.
    The scalar recurrence is an independent oracle in the unit tests.
    """
    if x.ndim != 4 or b.shape != c.shape or dt.shape != x.shape[:3]:
        raise ExperimentValidationError("invalid Mamba2 scan shapes")
    batch, time, heads, width = x.shape
    groups, state_dim = b.shape[-2:]
    if heads % groups or time < 1 or chunk_size < 1:
        raise ExperimentValidationError("invalid Mamba2 scan dimensions")
    x, dt, b, c = x.float(), dt.float(), b.float(), c.float()
    b = b.repeat_interleave(heads // groups, dim=2)
    c = c.repeat_interleave(heads // groups, dim=2)
    state = (x.new_zeros(batch, heads, width, state_dim) if initial is None else initial.float())
    outputs = []
    for start in range(0, time, chunk_size):
        end = min(time, start + chunk_size)
        xc, bc, cc = x[:, start:end], b[:, start:end], c[:, start:end]
        step = dt[:, start:end]
        log_decay = step * a.float()
        cumulative = log_decay.cumsum(1)
        # exp(sum_{k=j+1}^i dt_k A); upper triangle is masked BEFORE exp.
        diff = cumulative.transpose(1, 2).unsqueeze(-1) - cumulative.transpose(1, 2).unsqueeze(-2)
        causal = torch.ones(end - start, end - start, dtype=torch.bool, device=x.device).tril()
        decay = diff.masked_fill(~causal, -torch.inf).exp()
        cb = torch.einsum("bihn,bjhn->bhij", cc, bc)
        y = torch.einsum("bhij,bjhp->bihp", cb * decay, xc * step.unsqueeze(-1))
        y = y + torch.einsum("bihn,bhpn->bihp", cc, state) * cumulative.exp().unsqueeze(-1)
        outputs.append(y + xc * d.float().view(1, 1, heads, 1))
        tail = (cumulative[:, -1:] - cumulative).exp()
        state = state * cumulative[:, -1].exp().unsqueeze(-1).unsqueeze(-1)
        state = state + torch.einsum("bihp,bihn->bhpn", xc * (step * tail).unsqueeze(-1), bc)
    return torch.cat(outputs, dim=1), state


class _OffloadedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cpu_weight):
        ctx.weight = cpu_weight
        ctx.input_dtype = x.dtype
        return F.linear(x, cpu_weight.to(x.device, dtype=x.dtype))

    @staticmethod
    def backward(ctx, grad):
        weight = ctx.weight.to(grad.device, dtype=ctx.input_dtype)
        return torch.matmul(grad, weight), None


class FrozenLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, *, precision: str, device: str):
        super().__init__()
        self.offload = precision == "bf16_cpu_offload"
        if precision == "nf4":
            import bitsandbytes as bnb
            self.linear = bnb.nn.Linear4bit(weight.shape[1], weight.shape[0], bias=False,
                                           compute_dtype=torch.bfloat16, quant_type="nf4",
                                           compress_statistics=True)
            self.linear.weight = bnb.nn.Params4bit(weight.clone(), requires_grad=False,
                                                   quant_type="nf4", compress_statistics=True)
            self.linear.to(device)
            self.linear.requires_grad_(False)
        else:
            self.register_buffer("weight", weight.to(device="cpu" if self.offload else device,
                                                      dtype=torch.bfloat16).clone())

    def forward(self, x):
        if hasattr(self, "linear"):
            return self.linear(x)
        if self.offload:
            return _OffloadedLinear.apply(x, self.weight)
        return F.linear(x, self.weight)


class FrozenBlock(nn.Module):
    def __init__(self, config: dict, kind: str, take, *, precision: str, device: str):
        super().__init__()
        self.config, self.kind = config, kind
        self.eps = float(config["layer_norm_epsilon"])
        self.register_buffer("norm", take("norm.weight").to(device, torch.float32).clone())
        names = {"M": ("in_proj", "out_proj"), "*": ("q_proj", "k_proj", "v_proj", "o_proj"),
                 "-": ("up_proj", "down_proj")}[kind]
        self.linears = nn.ModuleDict({name: FrozenLinear(take(f"mixer.{name}.weight"),
                                                        precision=precision, device=device) for name in names})
        if kind == "M":
            for name, key in [("conv_weight", "conv1d.weight"), ("conv_bias", "conv1d.bias"),
                              ("dt_bias", "dt_bias"), ("a_log", "A_log"), ("d", "D"),
                              ("gated_norm", "norm.weight")]:
                self.register_buffer(name, take(f"mixer.{key}").to(device, torch.float32).clone())

    def forward(self, x, prefix=None):
        residual = x
        y = rms_norm(x, self.norm, self.eps)
        state = None
        cfg = self.config
        if self.kind == "-":
            y = self.linears["down_proj"](F.relu(self.linears["up_proj"](y)).square())
        elif self.kind == "*":
            batch, time, _ = y.shape
            heads, kvheads, width = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
            q = self.linears["q_proj"](y).view(batch, time, heads, width).transpose(1, 2)
            k = self.linears["k_proj"](y).view(batch, time, kvheads, width).transpose(1, 2)
            v = self.linears["v_proj"](y).view(batch, time, kvheads, width).transpose(1, 2)
            past = 0
            if prefix is not None:
                pk, pv = prefix
                past = pk.shape[2]
                k, v = torch.cat([pk, k], dim=2), torch.cat([pv, v], dim=2)
            state = (k, v)
            mask = torch.arange(k.shape[2], device=x.device)[None, :] <= (past + torch.arange(time, device=x.device))[:, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            y = self.linears["o_proj"](y.transpose(1, 2).reshape(batch, time, heads * width))
        else:
            heads, width, groups, size = cfg["mamba_num_heads"], cfg["mamba_head_dim"], cfg["n_groups"], cfg["ssm_state_size"]
            inner = heads * width
            z, xbc, dt = self.linears["in_proj"](y).split([inner, inner + 2 * groups * size, heads], dim=-1)
            history = xbc.new_zeros(xbc.shape[0], cfg["conv_kernel"] - 1, xbc.shape[-1]) if prefix is None else prefix[0]
            combined = torch.cat([history, xbc], dim=1)
            conv = F.conv1d(combined.transpose(1, 2), self.conv_weight.to(x.dtype),
                            self.conv_bias.to(x.dtype), groups=xbc.shape[-1]).transpose(1, 2)
            xc, b, c = F.silu(conv).split([inner, groups * size, groups * size], dim=-1)
            batch, time, _ = xc.shape
            dt = F.softplus(dt.float() + self.dt_bias)
            scan, ssm = ssd_scan(xc.view(batch, time, heads, width), dt, -self.a_log.exp(),
                                 b.view(batch, time, groups, size), c.view(batch, time, groups, size),
                                 self.d, None if prefix is None else prefix[1])
            gated = scan.reshape(batch, time, inner) * F.silu(z.float())
            y = rms_norm(gated, self.gated_norm, self.eps, groups=groups).to(x.dtype)
            y = self.linears["out_proj"](y)
            state = (combined[:, -(cfg["conv_kernel"] - 1):], ssm)
        return residual + y, state


def detach_state(value):
    if value is None:
        return None
    return tuple(x.detach().clone() for x in value)


class FrozenVoiceChatLM(nn.Module):
    """Only input gradients are computed; no parameter of this model is trained."""

    def __init__(self, checkpoint_path: Path, config_path: Path, *, precision: str = "nf4", device: str = "cuda"):
        super().__init__()
        from safetensors import safe_open

        if precision not in {"nf4", "bf16", "bf16_cpu_offload"}:
            raise ExperimentValidationError(f"unknown fitting precision {precision}")
        self.config = json.loads(config_path.read_text())
        self.precision, self.device_name = precision, device
        self.used_keys = set()
        with safe_open(checkpoint_path, framework="pt", device="cpu") as source:
            def take(key):
                self.used_keys.add(key)
                tensor = source.get_tensor(key)
                if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ExperimentValidationError(f"LM requires finite original F32 tensor: {key}")
                return tensor
            self.register_buffer("embedding", take("stt_model.embed_tokens.weight").to(torch.bfloat16).clone())
            self.head = FrozenLinear(take("stt_model.lm_head.weight"), precision=precision, device=device)
            self.function_head = FrozenLinear(take("stt_model.function_head.weight"), precision=precision, device=device)
            self.layers = nn.ModuleList()
            for index, kind in enumerate(self.config["hybrid_override_pattern"]):
                stem = f"stt_model.llm.layers.{index}."
                self.layers.append(FrozenBlock(self.config, kind, lambda key: take(stem + key),
                                               precision=precision, device=device))
            self.register_buffer("norm", take("stt_model.llm.norm_f.weight").to(device, torch.float32).clone())
            expected = {k for k in source.keys() if k.startswith("stt_model.llm.")}
            expected.update({"stt_model.embed_tokens.weight", "stt_model.lm_head.weight", "stt_model.function_head.weight"})
            if expected != self.used_keys:
                raise ExperimentValidationError("unconsumed or unexpected STT LM tensors")
        self.requires_grad_(False)
        self.eval()

    def embed(self, ids: torch.Tensor):
        return F.embedding(ids.cpu(), self.embedding).to(self.device_name)

    def forward(self, inputs_embeds, *, prefix=None, return_state=False, checkpoint_blocks=True):
        x = inputs_embeds.to(torch.bfloat16)
        states = []
        for index, layer in enumerate(self.layers):
            past = None if prefix is None else prefix[index]
            if checkpoint_blocks and torch.is_grad_enabled() and x.requires_grad:
                # Bind the current layer/cache: backward runs after the loop.
                def run(value, block=layer, cache=past):
                    return block(value, cache)[0]
                x = checkpoint(run, x, use_reentrant=False)
            else:
                x, state = layer(x, past)
                if return_state:
                    states.append(detach_state(state))
        x = rms_norm(x, self.norm, float(self.config["layer_norm_epsilon"]))
        return (x, states) if return_state else x

    @torch.no_grad()
    def cache_prompt(self, token_ids: list[int]):
        ids = torch.tensor([token_ids], dtype=torch.long)
        # The text channel starts from BOS; after the first conditioning frame
        # both output channels remain at pad (voicechat-cli.cpp:run_system).
        previous_text = torch.full_like(ids, 12)
        previous_text[:, 0] = 1
        pad = self.embed(torch.tensor([[12]]))
        _, states = self(self.embed(ids) + self.embed(previous_text) + pad, return_state=True)
        return states

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.parameters()):
            raise ExperimentValidationError("a language-model parameter is trainable or has a gradient")
