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


#: Which cached frame a streamed frame lines up with.
#:
#: The runtime streams one 80 ms block at a time and its encoder emits exactly
#: one frame per block.  The cache runs the same waveform in a single pass and
#: `features.frames_out` returns one frame MORE than there are blocks, so one
#: of them has no counterpart in the stream.
#:
#: The frontend settles which.  Each of the three stride-2 convolutions pads
#: two mel frames left and one right, so its output ``j`` reads inputs
#: ``2j-2 .. 2j``; composed three times, cached frame ``j`` reaches mel frame
#: ``8j``, i.e. audio through sample ``1280 * j`` -- the end of block ``j-1``.
#: The frame a streamer would emit having just consumed block ``i`` therefore
#: lines up with cached frame ``i + 1``, and cached frame 0 is the one the
#: stream never emits.
#:
#: Two probes agree without pinning it further, because the encoder attends
#: globally and no cached frame is a function of local audio alone: on silence
#: frames 0 and 1 are the run's largest outliers (cosine 0.17 against the
#: steady state, against 0.70 by frame 2), and a one-block burst perturbs three
#: to four frames from about ``L`` to ``L+3`` with a peak that wanders.  What
#: is left is half a frame of ambiguity -- 40 ms either way -- so this stays a
#: named constant that the fit can be re-run against.  See the frame-alignment
#: entry in `COMPARISON_7_RUN_LOG.md`.
CACHE_FRAME_OFFSET = 1


def duplex_audio_indices(frames: int, *, start: int = 0,
                         offset: int = CACHE_FRAME_OFFSET) -> list[int]:
    """A contiguous window into a silence-padded cache run.

    Every frame of a duplex timeline has an encoder frame behind it -- first
    the command, then the encoded silence the Realtime bridge keeps feeding on
    an input underrun -- so these indices are contiguous and never negative.
    `start` skips the cache's lead silence for a pool whose timeline begins at
    the command rather than at the bridge's first frame.
    """
    if frames < 1 or start < 0:
        raise ExperimentValidationError("a duplex timeline needs at least one audio frame")
    return list(range(offset + start, offset + start + frames))


def text_target_timeline(reply_ids: Sequence[int], audio_frames: int, *,
                         lead_frames: int = 0,
                         offset: int = CACHE_FRAME_OFFSET) -> dict[str, Any]:
    """B2's explicitly recorded schedule, at one token per 80 ms frame.

    Chat-format boundary tokens never enter a student's perception timeline.
    Speech frames supervise listening/pad; the reply starts after the audio.
    Silence after EOS trains the return to listening, rather than an arbitrary
    target-length sequence stop. B1 instead preserves the teacher's frame trace.

    The reply frames are not blank: they index the cache's encoded silence, so
    the student hears what deployment feeds it while it speaks.
    """
    if audio_frames < 1 or not reply_ids or any(token in {PAD, BOS, EOS, 10, 11} for token in reply_ids):
        raise ExperimentValidationError("B2 requires audio and a reply without channel boundary tokens")
    text = [PAD] * audio_frames + [BOS, *reply_ids, EOS] + [PAD] * 10
    return {"text_tokens": text, "function_tokens": [PAD] * len(text),
            "audio_indices": duplex_audio_indices(len(text), start=lead_frames, offset=offset),
            "schedule": "B2: listen through full audio, BOS, native-chat reply, EOS, 10 pad frames, "
                        "on encoded silence throughout"}


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


#: A duplex turn may open no earlier than this many frames before the streaming
#: encoder has consumed the command.  One or two frames is boundary slop -- the
#: encoder has its own startup latency and the last block is zero padded -- and
#: those replies are complete and on topic.  An opening well inside the command
#: is barge-in: real FT_EN behaviour, but not something to supervise here, since
#: it would teach the student to answer before it has heard the request.
DUPLEX_ONSET_TOLERANCE_FRAMES = 4
#: How long the teacher may take to open its turn.  The frozen FT_EN control
#: answers about 1.4 s past the audio in the deployment pilot.
DUPLEX_MAX_ONSET_FRAMES = 50


def parse_duplex_trace(trace: str, *, prefix_frames: int) -> dict[str, Any]:
    """The teacher's own per-frame trace, with nothing forced.

    Unlike `parse_audio_trace` this keeps the function channel as traced rather
    than rejecting spontaneous activity, and it records no audio indices: the
    duplex timeline has an encoder frame at every position -- the command, then
    encoded silence -- so how it maps onto a cached encoder run is settled when
    the cache is built, not here.
    """
    frames = re.findall(r"DUMP t=\s*(\d+).*?txt=\s*(\d+).*?fn=\s*(\d+)", trace)
    frames = [(int(t), int(txt), int(fn)) for t, txt, fn in frames if int(t) >= prefix_frames]
    # One trace file carries the whole session and the runtime's stderr is
    # block buffered, so a slice taken by byte offset can open with frames the
    # PREVIOUS turn had not flushed yet -- about one target in seven, at every
    # cadence measured.  Those frames precede this turn's in the file, so the
    # turn is the LAST run that starts where the system prompt ended and then
    # counts up without a gap.  Dropping them recovers the target; a real hole
    # inside the turn still fails the contiguity check below.
    starts = [index for index, frame in enumerate(frames) if frame[0] == prefix_frames]
    if not starts:
        raise ExperimentValidationError("the duplex teacher trace never reaches the prompt's last frame")
    frames = frames[starts[-1]:]
    if [f[0] for f in frames] != list(range(prefix_frames, prefix_frames + len(frames))):
        raise ExperimentValidationError("incomplete or noncontiguous duplex teacher frame trace")
    return {"text_tokens": [txt for _, txt, _ in frames],
            "function_tokens": [fn for _, _, fn in frames],
            "schedule": "duplex: the teacher's own frame trace; the turn is opened by the model, "
                        "not by VC_FORCE_BOS, and every frame has an encoder frame behind it"}


def duplex_turn_rejections(*, opened: bool, onset: int | None, spoken: int | None,
                           tolerance: int = DUPLEX_ONSET_TOLERANCE_FRAMES,
                           max_onset: int = DUPLEX_MAX_ONSET_FRAMES) -> list[str]:
    """Why this duplex turn may not be supervised, if it may not."""
    rejections = []
    if not opened:
        rejections.append("never opened a turn")
        return rejections
    if onset is None:
        rejections.append("no measurable turn onset")
        return rejections
    if onset < -tolerance:
        rejections.append(f"barged in {-onset} frames before the command ended")
    elif onset > max_onset:
        rejections.append(f"took {onset} frames to open")
    if not spoken:
        rejections.append("opened a turn but said nothing")
    return rejections


def duplex_inputs(projection: nn.Linear, lm: Any, features: torch.Tensor, timeline: Mapping[str, Any]):
    """Same-frame CE: the model consumes the PREVIOUS output tokens plus audio."""
    text = torch.tensor(timeline["text_tokens"], dtype=torch.long)
    function = torch.tensor(timeline["function_tokens"], dtype=torch.long)
    indices = torch.tensor(timeline["audio_indices"], device=features.device)
    if text.ndim != 1 or text.numel() != function.numel() or text.numel() != indices.numel():
        raise ExperimentValidationError("duplex timeline arrays have different lengths")
    # Every frame hears something.  The v1 timelines instead marked the frames
    # past the command -1 and fed an exact zero embedding there, which is what
    # `run_turn` passes but not what the Realtime bridge does: it advances the
    # model on encoded silence, and a projection fitted against the zero waits
    # for a cue deployment never sends.  Such a timeline is not fittable.
    if indices.min() < 0:
        raise ExperimentValidationError(
            "duplex timeline has frames without audio; the tail must index encoded silence")
    if indices.max() >= features.shape[0]:
        raise ExperimentValidationError("duplex timeline indexes outside cached audio")
    audio = projection(features.float())[indices]
    previous_text = torch.cat([torch.tensor([PAD]), text[:-1]])
    previous_function = torch.cat([torch.tensor([PAD]), function[:-1]])
    inputs = lm.fuse(audio.unsqueeze(0), previous_text[None], previous_function[None])
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


@torch.no_grad()
def duplex_free_run(projection: nn.Linear, lm: Any, features: torch.Tensor,
                    timeline: Mapping[str, Any], prefix, *, command_frames: int,
                    max_reply_frames: int = 200,
                    max_onset_frames: int = DUPLEX_MAX_ONSET_FRAMES) -> dict[str, Any]:
    """Decode the turn with nothing forced, the way `duplex_step` runs it.

    `token_loss` is teacher forced: every frame is scored against the teacher's
    token having been fed the teacher's history.  Deployment gives the model
    none of that, and three of the differences are exactly the ones that sank
    the v1 pilot -- no opening BOS is forced, no barge-in is suppressed, and
    after the command the model hears encoded silence rather than a zero
    embedding.  This reproduces all three and reports when the turn opened.

    Only the text channel is decoded.  The checkpoint's separate function head
    is not modelled here, so the function channel is fed pad, which is what the
    Realtime bridge feeds whenever it splices nothing.
    """
    indices = torch.tensor(timeline["audio_indices"], device=features.device)
    if indices.numel() < 1 or int(indices.min()) < 0 or int(indices.max()) >= features.shape[0]:
        raise ExperimentValidationError("free-running timeline indexes outside cached audio")
    if not 0 < command_frames <= indices.numel():
        raise ExperimentValidationError("the command must end inside the free-running timeline")
    audio = projection(features.float())[indices]
    pad = torch.tensor([[PAD]])
    state, tokens, opened_at = prefix, [], None
    # `cache_prompt` leaves both output channels at pad after conditioning, so
    # the first audio frame consumes a pad exactly as `duplex_inputs` does.
    previous = PAD
    for step in range(audio.shape[0]):
        inputs = lm.fuse(audio[step].view(1, 1, -1), torch.tensor([[previous]]), pad)
        hidden, state = lm(inputs, prefix=state, return_state=True)
        token = int(lm.head(hidden[:, -1]).float().argmax(-1))
        tokens.append(token)
        # A pad keeps listening; an EOS with no turn open closes nothing.  BOS
        # or any content token is the model deciding to speak.
        if opened_at is None and token not in (PAD, EOS):
            opened_at = step
        elif opened_at is not None and token == EOS:
            break
        if opened_at is not None and step - opened_at >= max_reply_frames:
            break
        # A turn that has not opened this far past the command is one the
        # teacher filter would have rejected anyway, so decoding the rest of
        # the silence tail only makes the silent cases the expensive ones.
        if opened_at is None and step >= command_frames + max_onset_frames:
            break
        previous = token
    spoken = [t for t in tokens[opened_at:] if t not in (PAD, BOS, EOS)] if opened_at is not None else []
    return {"opened": opened_at is not None,
            "onset_frames_past_command": None if opened_at is None else opened_at - command_frames,
            "frames_decoded": len(tokens), "text_tokens": tokens, "reply_token_ids": spoken,
            "conditions": ["no forced BOS", "no barge-in suppression", "encoded silence after the command"]}


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


#: How far the fitted projection's free-running open rate may fall below the
#: untouched `FT_EN` projection's before the fit is presumed to have broken the
#: turn taking deployment needs.  The v1 pilot answered 0 of 24 where its own
#: initialization answered 24 of 24, so anything but a near-tie is a finding.
DUPLEX_OPEN_RATE_TOLERANCE = 0.05


def duplex_gate_verdict(fitted: Mapping[str, Any], initialization: Mapping[str, Any], *,
                        tolerance: float = DUPLEX_OPEN_RATE_TOLERANCE,
                        onset_tolerance: int = DUPLEX_ONSET_TOLERANCE_FRAMES,
                        max_onset: int = DUPLEX_MAX_ONSET_FRAMES) -> dict[str, Any]:
    """Did fitting cost the projection its ability to open a turn at all?

    `english_gate_verdict` scores teacher-forced cross-entropy, which the v1
    projection passed comfortably while answering nothing in deployment: fed
    the teacher's own history it never had to decide to speak.  This scores the
    decision itself, free running under the three deployment conditions, and
    the untouched `FT_EN` projection is the control that says the harness works.
    """

    import statistics

    if not fitted or set(fitted) != set(initialization):
        raise ExperimentValidationError("the duplex gate needs the same cells before and after fitting")
    cells: dict[str, Any] = {}
    control_opened = control_total = fitted_opened = total = 0
    for cell, after in sorted(fitted.items()):
        before = initialization[cell]
        if int(after["n"]) != int(before["n"]) or int(after["n"]) < 1:
            raise ExperimentValidationError(f"duplex gate cell {cell} changed size")
        rate_after = int(after["opened"]) / int(after["n"])
        rate_before = int(before["opened"]) / int(before["n"])
        onsets = [int(o) for o in after.get("onsets", [])]
        in_band = [o for o in onsets if -onset_tolerance <= o <= max_onset]
        cells[cell] = {
            "n": int(after["n"]),
            "initialization_open_rate": rate_before, "fitted_open_rate": rate_after,
            "delta": rate_after - rate_before,
            "initialization_onset_median": (statistics.median(before["onsets"]) if before.get("onsets") else None),
            "fitted_onset_median": (statistics.median(onsets) if onsets else None),
            "fitted_onsets_in_band": len(in_band),
            "passed": bool(rate_after >= rate_before - tolerance)}
        control_opened += int(before["opened"])
        control_total += int(before["n"])
        fitted_opened += int(after["opened"])
        total += int(after["n"])
    # If the untouched projection cannot open a turn either, the harness is
    # wrong and a fitted pass would mean nothing.
    calibrated = control_opened == control_total
    return {"rule": ("free running with no forced BOS, no barge-in suppression and encoded silence "
                     "after the command, the fitted projection must open a turn at a rate within "
                     f"{tolerance} of the untouched FT_EN projection in every prompt cell"),
            "tolerance": float(tolerance), "cells": cells, "clips_scored": total,
            "initialization_open_rate": control_opened / control_total,
            "fitted_open_rate": fitted_opened / total,
            "control_calibrated": calibrated,
            "control_rule": "the untouched FT_EN projection must open every held-out turn",
            "certifies": "the deployment turn decision, which teacher-forced cross-entropy cannot see",
            "passed": bool(calibrated and all(cell["passed"] for cell in cells.values()))}


def candidate_provenance(*, arm: str, budget: int, manifest: Mapping[str, Any],
                         source: Mapping[str, Any], initialization: Mapping[str, Any],
                         language_model: Mapping[str, Any], targets: Mapping[str, Any],
                         calibration: Mapping[str, Any], fitting_graph: Mapping[str, Any],
                         supervision: Mapping[str, Any]) -> dict[str, Any]:
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
    # Where the timeline reads the cache is a free choice with a measured but
    # not fully pinned answer, so a fit that used another one is a different
    # candidate and has to hash differently.
    if "frame_offset" not in supervision or "activation_cache" not in supervision:
        raise ExperimentValidationError("a fit must record its frame alignment and activation cache")
    value = {"comparison": COMPARISON, "artifact_kind": ARTIFACT_KIND, "arm": arm,
             "data_budget_percent": budget, "arm_definition": dataset_b.ARMS[arm],
             "encoder_source": dict(source), "initialization": dict(initialization),
             "training_manifest_sha256": manifest["manifest_sha256"],
             "budget_clip_ids": dataset_b.budget_clip_ids(manifest, budget),
             "system_prompts": dict(gating.SYSTEM_PROMPTS),
             "teacher_paths": dict(manifest["teacher_paths"]), "targets": dict(targets),
             "language_model": dict(language_model), "loss_calibration": dict(calibration),
             "fitting_graph": dict(fitting_graph), "supervision": dict(supervision),
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
