"""Comparison 7 supervision, frozen provenance and projection-only fitting."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from . import dataset_b, gating
from .experiments import ExperimentValidationError, stable_json_sha256

PAD, BOS, EOS = 12, 1, 2
COMPARISON = 7
ARTIFACT_KIND = "gradient_fitted_interface"
#: How much held-out English token cross-entropy E1 may lose to the shared B2
#: term before the training loop is presumed broken rather than merely traded
#: off.  Deliberately tight: E1 begins at its own teacher's answer.
GATE_TOLERANCE_NATS = 0.05


def text_target_timeline(reply_ids: Sequence[int], audio_frames: int) -> dict[str, Any]:
    """B2's explicitly recorded schedule, at one token per 80 ms frame.

    Chat-format boundary tokens never enter a student's perception timeline.
    Speech frames supervise listening/pad; the reply starts after the audio.
    Silence after EOS trains the return to listening, rather than an arbitrary
    target-length sequence stop. B1 instead preserves the teacher's frame trace.
    """
    if audio_frames < 1 or not reply_ids or any(token in {PAD, BOS, EOS, 10, 11} for token in reply_ids):
        raise ExperimentValidationError("B2 requires audio and a reply without channel boundary tokens")
    text = [PAD] * audio_frames + [BOS, *reply_ids, EOS] + [PAD] * 10
    return {"text_tokens": text, "function_tokens": [PAD] * len(text),
            "audio_indices": list(range(audio_frames)) + [-1] * (len(text) - audio_frames),
            "schedule": "B2: listen through full audio, BOS, native-chat reply, EOS, 10 pad frames"}


def parse_audio_trace(trace: str, *, prefix_frames: int, audio_frames: int) -> dict[str, Any]:
    frames = re.findall(r"DUMP t=\s*(\d+).*?txt=\s*(\d+).*?fn=\s*(\d+)", trace)
    frames = [(int(t), int(txt), int(fn)) for t, txt, fn in frames if int(t) >= prefix_frames]
    if not frames or [f[0] for f in frames] != list(range(prefix_frames, prefix_frames + len(frames))):
        raise ExperimentValidationError("incomplete or noncontiguous B1 teacher frame trace")
    if len(frames) < audio_frames:
        raise ExperimentValidationError("B1 teacher ended before consuming the whole command")
    if any(fn != PAD for _, _, fn in frames):
        raise ExperimentValidationError("B1 spontaneous function activity requires a traced audio-splice schedule")
    return {"text_tokens": [txt for _, txt, _ in frames],
            "function_tokens": [fn for _, _, fn in frames],
            "audio_indices": list(range(audio_frames)) + [-1] * (len(frames) - audio_frames),
            "schedule": "B1: exact original VoiceChat per-frame teacher trace; no function splices"}


def duplex_inputs(projection: nn.Linear, lm: Any, features: torch.Tensor, timeline: Mapping[str, Any]):
    """Same-frame CE: the model consumes the PREVIOUS output tokens plus audio."""
    text = torch.tensor(timeline["text_tokens"], dtype=torch.long)
    function = torch.tensor(timeline["function_tokens"], dtype=torch.long)
    indices = torch.tensor(timeline["audio_indices"], device=features.device)
    if text.ndim != 1 or text.numel() != function.numel() or text.numel() != indices.numel():
        raise ExperimentValidationError("duplex timeline arrays have different lengths")
    if indices.min() < -1 or indices.max() >= features.shape[0]:
        raise ExperimentValidationError("duplex timeline indexes outside cached audio")
    projected = projection(features.float())
    # Zero embedding past the audio means zero AFTER proj, including its bias.
    audio = projected[indices.clamp_min(0)] * (indices >= 0).unsqueeze(-1)
    previous_text = torch.cat([torch.tensor([PAD]), text[:-1]])
    previous_function = torch.cat([torch.tensor([PAD]), function[:-1]])
    inputs = audio.unsqueeze(0) + lm.embed(previous_text[None]) + lm.embed(previous_function[None])
    return inputs, text.to(features.device)


def token_loss(projection: nn.Linear, lm: Any, features: torch.Tensor, timeline: Mapping[str, Any], prefix):
    inputs, labels = duplex_inputs(projection, lm, features, timeline)
    hidden = lm(inputs, prefix=prefix)
    # Chunk the large vocabulary head; the mean is per token, independent of
    # reply length. No extra HF causal shift: targets are already frame aligned.
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, len(labels), 16):
        logits = lm.head(hidden[:, start:start + 16]).float()
        total = total + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[start:start + 16], reduction="sum")
    return total / len(labels)


def calibrate_pool_weights(norms: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Freeze an E1 training-only gradient-scale calibration shared by all arms."""
    if set(norms) != {"B1", "B2"}:
        raise ExperimentValidationError("loss calibration requires both supervision pools")
    medians = {}
    for pool, values in norms.items():
        if not values or any(not math.isfinite(x) or x <= 0 for x in values):
            raise ExperimentValidationError("loss calibration requires positive finite gradient norms")
        medians[pool] = float(torch.tensor(values, dtype=torch.float64).median())
    raw = {pool: 1 / value for pool, value in medians.items()}
    weights = {pool: 2 * value / sum(raw.values()) for pool, value in raw.items()}
    return {"method": "inverse median E1 initial projection gradient norm; mean weight one",
            "source_split": "Dataset B train calibration subset", "median_gradient_norms": medians,
            "weights": weights, "raw_norms": {k: list(v) for k, v in norms.items()}}


def candidate_provenance(*, arm: str, budget: int, manifest: Mapping[str, Any],
                         source: Mapping[str, Any], initialization: Mapping[str, Any],
                         language_model: Mapping[str, Any], targets: Mapping[str, Any],
                         calibration: Mapping[str, Any]) -> dict[str, Any]:
    dataset_b.validate_manifest(manifest)
    if arm not in dataset_b.ARMS or budget not in dataset_b.BUDGETS:
        raise ExperimentValidationError("unknown Comparison 7 arm or data budget")
    if language_model.get("fitting_precision") not in {"nf4", "bf16", "bf16_cpu_offload"}:
        raise ExperimentValidationError("fitting LM precision must be explicit")
    for name, record in [("source", source), ("initialization", initialization), ("language_model", language_model)]:
        if not record.get("sha256"):
            raise ExperimentValidationError(f"missing {name} content hash")
    if targets.get("teacher_paths") != manifest["teacher_paths"] or not targets.get("manifest_sha256"):
        raise ExperimentValidationError("target teacher path or content hash is missing or mismatched")
    value = {"comparison": COMPARISON, "artifact_kind": ARTIFACT_KIND, "arm": arm,
             "data_budget_percent": budget, "arm_definition": dataset_b.ARMS[arm],
             "encoder_source": dict(source), "initialization": dict(initialization),
             "training_manifest_sha256": manifest["manifest_sha256"],
             "budget_clip_ids": dataset_b.budget_clip_ids(manifest, budget),
             "system_prompts": dict(gating.SYSTEM_PROMPTS),
             "teacher_paths": dict(manifest["teacher_paths"]), "targets": dict(targets),
             "language_model": dict(language_model), "loss_calibration": dict(calibration),
             "trainable_tensors": ["proj.weight", "proj.bias"],
             "precision_stage": "pre_quantization"}
    value["provenance_sha256"] = stable_json_sha256(value)
    return value


def english_gate_verdict(fitted: Mapping[str, Any], initialization: Mapping[str, Any], *,
                         tolerance: float = GATE_TOLERANCE_NATS) -> dict[str, Any]:
    """Comparison 7's pipeline check, in the design record's §9 terms.

    E1 trains `FT_EN`'s own projection against targets `FT_EN` itself produced,
    so on English it starts at the answer.  If fitting moves it materially away
    from where it began, the loop is broken and no later arm means anything.
    That is what this measures: held-out English token cross-entropy under the
    fitted projection against the same clips under the initialization, one cell
    per prompt condition, so a regression cannot hide inside an average.

    It certifies the loop, not the deployment.  The `FT_EN` control row of the
    speech-to-action table is a separate and later check under invariant 8.
    """

    if not fitted or set(fitted) != set(initialization):
        raise ExperimentValidationError("the English gate needs the same cells before and after fitting")
    cells: dict[str, Any] = {}
    weighted = {"fitted": 0.0, "initialization": 0.0}
    total = 0
    for cell, after in sorted(fitted.items()):
        before = initialization[cell]
        if int(after["n"]) != int(before["n"]) or int(after["n"]) < 1:
            raise ExperimentValidationError(f"English gate cell {cell} changed size")
        if not math.isfinite(float(after["mean"])) or not math.isfinite(float(before["mean"])):
            raise ExperimentValidationError(f"non-finite English gate loss in {cell}")
        delta = float(after["mean"]) - float(before["mean"])
        cells[cell] = {"n": int(after["n"]), "initialization_token_ce": float(before["mean"]),
                       "fitted_token_ce": float(after["mean"]), "delta": delta,
                       "passed": bool(delta <= tolerance)}
        total += int(after["n"])
        weighted["fitted"] += float(after["mean"]) * int(after["n"])
        weighted["initialization"] += float(before["mean"]) * int(before["n"])
    return {"rule": ("held-out English token cross-entropy under the fitted projection must not "
                     f"exceed the initialization by more than {tolerance} nats in any prompt cell"),
            "tolerance_nats": float(tolerance), "cells": cells, "clips_scored": total,
            "initialization_token_ce": weighted["initialization"] / total,
            "fitted_token_ce": weighted["fitted"] / total,
            "delta": (weighted["fitted"] - weighted["initialization"]) / total,
            "certifies": "the fitting loop on English; not the deployment endpoint",
            "passed": all(cell["passed"] for cell in cells.values())}
