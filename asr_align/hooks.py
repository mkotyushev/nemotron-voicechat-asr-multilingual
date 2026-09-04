"""Where to measure a FastConformer, and what ties its widths together.

The architecture decides the grouping, exactly as it does in `ot_width.py`:

``residual``
    One plan for the whole 1024-wide residual stream. Per-block plans are not
    available: `x + f(x)` forces every block boundary to share one basis, and
    there are twenty-five of them here -- the subsampling output and each block's
    `norm_out`.

``block.{i}.ffn1`` / ``block.{i}.ffn2``
    The 4096-wide inner width of each macaron feed forward. Nothing ties one
    block's to another's.

``block.{i}.conv``
    The 1024-wide channel space between the GLU and the depthwise convolution.
    The depthwise convolution is diagonal in it, so a permutation there is free;
    anything denser would have to move the convolution kernel too.

``block.{i}.head``
    Eight units, one per attention head, expanded by the head dimension when
    applied. Heads have no canonical internal basis, so they are compared
    through the only representation both models agree on -- the vector each head
    adds to the residual stream -- which needs the residual plan first. That is
    why this is a second pass.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Iterable

import torch
from torch import nn

from .encoder import VoiceChatPerception
from .transport import (
    AlignmentStatistics,
    TransportOperators,
    as_unit_matrix,
)

RESIDUAL_GROUP = "residual"
GROUP_KINDS = ("residual", "ffn", "conv", "head")


def _store(module, args, output, *, store: dict, key: str, transform=None) -> None:
    store[key] = output if transform is None else transform(output)


def _store_input(module, args, *, store: dict, key: str, transform=None) -> None:
    store[key] = args[0] if transform is None else transform(args[0])


def register(model: VoiceChatPerception, store: dict, kinds: Iterable[str]) -> list:
    """Hooks for every group in ``kinds`` except ``head``, which needs two passes."""

    kinds = set(kinds)
    handles = []
    if RESIDUAL_GROUP in kinds:
        handles.append(
            model.encoder.subsampling.register_forward_hook(
                partial(_store, store=store, key=f"{RESIDUAL_GROUP}.subsampling")
            )
        )
    for index, block in enumerate(model.encoder.layers):
        if RESIDUAL_GROUP in kinds:
            handles.append(
                block.register_forward_hook(
                    partial(_store, store=store, key=f"{RESIDUAL_GROUP}.block.{index}")
                )
            )
        if "ffn" in kinds:
            for which in (1, 2):
                feed_forward = getattr(block, f"feed_forward{which}")
                handles.append(
                    feed_forward.linear1.register_forward_hook(
                        partial(
                            _store,
                            store=store,
                            key=f"block.{index}.ffn{which}",
                            transform=torch.nn.functional.silu,
                        )
                    )
                )
        if "conv" in kinds:
            # the depthwise convolution's input is the GLU output, laid out
            # (batch, channels, frames); every other hook point is frames-last
            handles.append(
                block.conv.depthwise_conv.register_forward_pre_hook(
                    partial(
                        _store_input,
                        store=store,
                        key=f"block.{index}.conv",
                        transform=lambda x: x.transpose(1, 2),
                    )
                )
            )
    return handles


def register_heads(model: VoiceChatPerception, store: dict) -> list:
    return [
        block.self_attn.o_proj.register_forward_pre_hook(
            partial(_store_input, store=store, key=f"block.{index}.head")
        )
        for index, block in enumerate(model.encoder.layers)
    ]


def head_contributions(
    attention_output: torch.Tensor,
    output_weight: torch.Tensor,
    n_head: int,
) -> torch.Tensor:
    """Each head's additive contribution to the residual stream.

    `(heads, batch * frames, n_embd)`, directly comparable across two models once
    the residual axis is in a shared basis.
    """

    batch, frames, width = attention_output.shape
    d_head = width // n_head
    per_head = attention_output.reshape(batch, frames, n_head, d_head)
    blocks = output_weight.reshape(output_weight.shape[0], n_head, d_head)
    contributions = torch.einsum("bthd,ehd->hbte", per_head, blocks)
    return contributions.reshape(n_head, batch * frames, -1)


@torch.no_grad()
def collect(
    source: VoiceChatPerception,
    target: VoiceChatPerception,
    mels: Iterable[torch.Tensor],
    kinds: Iterable[str],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, AlignmentStatistics]:
    """First pass: residual, feed-forward and convolution cross-moments."""

    kinds = {kind for kind in kinds if kind != "head"}
    if not kinds:
        return {}
    source_store: dict[str, torch.Tensor] = {}
    target_store: dict[str, torch.Tensor] = {}
    handles = register(source, source_store, kinds) + register(target, target_store, kinds)
    statistics: dict[str, AlignmentStatistics] = {}
    try:
        for index, mel in enumerate(mels):
            source_store.clear()
            target_store.clear()
            source(mel)
            target(mel)
            for key, value in source_store.items():
                group = RESIDUAL_GROUP if key.startswith(RESIDUAL_GROUP) else key
                other = target_store[key]
                statistics.setdefault(
                    group, AlignmentStatistics(value.shape[-1], other.shape[-1])
                ).update(as_unit_matrix(value.float()), as_unit_matrix(other.float()))
            if progress is not None:
                progress(f"stream statistics: batch {index + 1}")
    finally:
        for handle in handles:
            handle.remove()
    return statistics


@torch.no_grad()
def collect_heads(
    source: VoiceChatPerception,
    target: VoiceChatPerception,
    mels: Iterable[torch.Tensor],
    residual: TransportOperators,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, AlignmentStatistics]:
    """Second pass: head cross-moments, in the residual basis the first pass found."""

    source_store: dict[str, torch.Tensor] = {}
    target_store: dict[str, torch.Tensor] = {}
    handles = register_heads(source, source_store) + register_heads(target, target_store)
    n_head_source = source.hyper.n_head
    n_head_target = target.hyper.n_head
    statistics: dict[str, AlignmentStatistics] = {}
    try:
        for index, mel in enumerate(mels):
            source_store.clear()
            target_store.clear()
            source(mel)
            target(mel)
            for key, value in source_store.items():
                block = int(key.split(".")[1])
                mine = residual.project_out(
                    head_contributions(
                        value.float(),
                        source.encoder.layers[block].self_attn.o_proj.weight.float(),
                        n_head_source,
                    ),
                    dim=-1,
                )
                theirs = head_contributions(
                    target_store[key].float(),
                    target.encoder.layers[block].self_attn.o_proj.weight.float(),
                    n_head_target,
                )
                statistics.setdefault(
                    key, AlignmentStatistics(n_head_source, n_head_target)
                ).update(mine.reshape(n_head_source, -1), theirs.reshape(n_head_target, -1))
            if progress is not None:
                progress(f"head statistics: batch {index + 1}")
    finally:
        for handle in handles:
            handle.remove()
    return statistics


def module_by_group(model: nn.Module, group: str) -> nn.Module:  # pragma: no cover - helper
    """The module a group's plan applies to, for callers that need to walk them."""

    if group == RESIDUAL_GROUP:
        return model.encoder
    parts = group.split(".")
    block = model.encoder.layers[int(parts[1])]
    if parts[2].startswith("ffn"):
        return getattr(block, f"feed_forward{parts[2][-1]}")
    if parts[2] == "conv":
        return block.conv
    return block.self_attn
