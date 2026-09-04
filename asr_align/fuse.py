"""Fusing the two encoders' weights, once transport says how they correspond.

The interface map in :mod:`asr_align.interface` leaves the swapped-in encoder
alone and corrects its output. This module does the other thing the paper is
actually about: it moves weights.

Three checkpoints here descend from one ancestor.
`nvidia/nemotron-speech-streaming-en-0.6b` (EN) is the published English
encoder; VoiceChat's `stt_model.perception` (VC) is a fine-tune of it, close
enough that `proj` still reads EN; and `nvidia/nemotron-3.5-asr-streaming-0.6b`
(ML) is a continued-training descendant of the same EN, far enough that `proj`
does not. So the multilingual capability is a *direction* in weight space,
`ML - EN`, that can be added to VC:

    W = W_VC + lambda * T(W_ML - W_EN)

`lambda = 0` is the encoder that works and speaks English; `lambda = 1` is
VoiceChat's fine-tune carrying the whole of the multilingual update. Somewhere
between is a Pareto front, and the interface map is refitted at each point, so
the two mechanisms compose rather than compete.

`T` is where the transport plans come in: a weight-space direction only means
something if the two checkpoints number their neurons the same way, and the
plans are the evidence for or against that. Where a group has no plan the
operators default to the identity, which makes this plain task arithmetic --
the right default if :func:`asr_align.transport.identity_agreement` comes back
at one, and a silent mistake if it does not, so the caller passes plans in
rather than this module deciding.

Every axis below is mapped by the group that owns it. A permutation plan makes
all of it exact; a barycentric plan averages neurons, which the depthwise
convolution and the GLU halves can only approximate, so `--column-assignment
integral` is the honest choice when fusing.
"""

from __future__ import annotations

import torch

from .experiments import candidate as strict_candidate
from .experiments import task_vector, validate_encoder_triplet
from .transport import TransportOperators, concatenated_operators, expanded_operators, identity_operators

RESIDUAL_GROUP = "residual"


def _get(
    operators: dict[str, TransportOperators], key: str, size: int
) -> TransportOperators:
    found = operators.get(key)
    return found if found is not None else identity_operators(size)


def rebase(
    state: dict[str, torch.Tensor],
    operators: dict[str, TransportOperators],
    *,
    n_layer: int,
    n_head: int,
) -> dict[str, torch.Tensor]:
    """Express one encoder's weights in another's basis.

    Only tensors whose axes belong to a group with a plan move. The subsampling
    stack's own 256-channel space has no plan -- it is upstream of the residual
    stream and nothing downstream reads it -- so those kernels are carried over
    index for index, and only `subsampling.linear`'s output axis is mapped.
    """

    out = dict(state)
    n_embd = state["encoder.subsampling.linear.weight"].shape[0]
    residual = _get(operators, RESIDUAL_GROUP, n_embd)
    d_head = n_embd // n_head

    out["encoder.subsampling.linear.weight"] = residual.project_out(
        state["encoder.subsampling.linear.weight"], 0
    )
    out["encoder.subsampling.linear.bias"] = residual.project_out(
        state["encoder.subsampling.linear.bias"], 0
    )

    for il in range(n_layer):
        p = f"encoder.layers.{il}."
        heads = expanded_operators(_get(operators, f"block.{il}.head", n_head), d_head)
        conv = _get(operators, f"block.{il}.conv", n_embd)
        glu = concatenated_operators([conv, conv])

        for norm in ("norm_self_att", "norm_out", "norm_feed_forward1", "norm_feed_forward2", "norm_conv"):
            for suffix in ("weight", "bias"):
                out[p + norm + "." + suffix] = residual.project_out(
                    state[p + norm + "." + suffix], 0
                )

        for name in ("q_proj", "k_proj", "v_proj"):
            key = p + f"self_attn.{name}.weight"
            out[key] = heads.project_out(residual.project_in(state[key], 1), 0)
        # the relative position projection reads a fixed sinusoid basis, not the
        # residual stream, so only its output axis belongs to a group
        key = p + "self_attn.relative_k_proj.weight"
        out[key] = heads.project_out(state[key], 0)
        key = p + "self_attn.o_proj.weight"
        out[key] = residual.project_out(heads.project_in(state[key], 1), 0)
        for name in ("bias_u", "bias_v"):
            key = p + f"self_attn.{name}"
            flat = state[key].reshape(-1)
            out[key] = heads.project_out(flat, 0).reshape(state[key].shape)

        for which in (1, 2):
            inner = _get(operators, f"block.{il}.ffn{which}", state[p + f"feed_forward{which}.linear1.weight"].shape[0])
            key = p + f"feed_forward{which}.linear1.weight"
            out[key] = inner.project_out(residual.project_in(state[key], 1), 0)
            key = p + f"feed_forward{which}.linear2.weight"
            out[key] = residual.project_out(inner.project_in(state[key], 1), 0)

        key = p + "conv.pointwise_conv1.weight"
        out[key] = glu.project_out(residual.project_in(state[key], 1), 0)
        key = p + "conv.depthwise_conv.weight"
        out[key] = conv.project_out(state[key], 0)
        for suffix in ("weight", "bias"):
            key = p + "conv.norm." + suffix
            out[key] = conv.project_out(state[key], 0)
        key = p + "conv.pointwise_conv2.weight"
        out[key] = residual.project_out(conv.project_in(state[key], 1), 0)

    return out


def task_arithmetic(
    base: dict[str, torch.Tensor],
    ancestor: dict[str, torch.Tensor],
    descendant: dict[str, torch.Tensor],
    weight: float,
) -> dict[str, torch.Tensor]:
    """`base + weight * (descendant - ancestor)`, tensor by tensor.

    All three must already be in one basis; run :func:`rebase` first if the
    transport plans say they are not.  Arithmetic is deliberately limited to
    canonical ``encoder.*`` tensors and delegated to the shared strict path,
    which rejects missing keys, shape mismatches, broadcasting and non-finite
    values and always computes in F32.
    """

    states = validate_encoder_triplet(ancestor, base, descendant)
    delta = task_vector(states["E"], states["F"])
    return strict_candidate(states["M"], delta, weight)


def interpolate(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    weight: float,
) -> dict[str, torch.Tensor]:
    """`(1 - weight) * left + weight * right`, the plain fusion of the paper."""

    return {key: (1.0 - weight) * left[key] + weight * right[key] for key in left}
