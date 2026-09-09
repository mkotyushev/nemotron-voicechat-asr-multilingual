#!/usr/bin/env python3
"""Comparison 7: freeze sources, cache encoders, generate targets and fit proj.

Each stage is resumable and content addressed. An unfinished stage never writes
a completed candidate or marks an experiment checklist item complete.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from asr_align import baseline, dataset_b, encoder, export, features, final_map, gating, interface_fit, manifests, weights
from asr_align.experiments import (
    ExperimentValidationError,
    assert_runtime_config_inherited,
    sha256_file,
    stable_json_sha256,
)
from asr_align.frozen_lm import FITTING_GRAPH_VERSION, FrozenVoiceChatLM, fusion_weights_from_config
from asr_align.interface_fit import (
    candidate_provenance,
    calibrate_pool_weights,
    duplex_turn_rejections,
    english_gate_verdict,
    parse_duplex_trace,
    text_target_timeline,
    token_loss,
)

LOG = logging.getLogger("interface-fitting")


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_frozen(path: Path) -> dict:
    value = manifests.load_manifest(path)
    return value


def write_frozen(path: Path, value: dict):
    return manifests.write_frozen(path, {"schema_version": "1.0", **value})


def verify_record(value: dict):
    path = Path(value["path"])
    if not path.is_file() or path.stat().st_size != value["bytes"] or sha256_file(path) != value["sha256"]:
        raise ExperimentValidationError(f"changed source: {path}")


def prepare(args):
    setup = baseline.load_shared_setup(args.shared_setup)
    data = read_frozen(args.dataset_b)
    dataset_b.verify_audio(data)
    run6 = json.loads((args.comparison_6 / "run.json").read_text())
    # load_shared_setup has already hashed the original checkpoints. Reuse its
    # verified records rather than rereading the 44 GB source for each arm.
    sources = {arm: next(dict(f) for f in setup.value["checkpoints"][role]["files"]
                         if Path(f["path"]).name == "model.safetensors")
               for arm, role in [("E1", "F"), ("E2", "M")]}
    for arm, name in [("E3", "regmean-plus-plus"), ("E4", "simple-average")]:
        art = run6["candidates"][name]["artifact"]
        source = args.comparison_6 / art["path"] / "model.safetensors"
        expected = next(f for f in art["files"] if f["path"] == "model.safetensors")
        sources[arm] = record(source)
        if any(sources[arm][k] != expected[k] for k in ("bytes", "sha256")):
            raise ExperimentValidationError(f"{arm} differs from the frozen Comparison 6 artifact")
    folded = args.comparison_3 / "artifacts/final-map-projection/model.safetensors"
    run3 = json.loads((args.comparison_3 / "run.json").read_text())
    # The complete run record is pinned as well as the initialization artifact.
    folded_record = record(folded)
    initializations = {arm: dict(folded_record if arm == "E2" else sources["E1"]) for arm in dataset_b.ARMS}
    voicechat_config = next(dict(f) for f in setup.value["checkpoints"]["F"]["files"]
                           if Path(f["path"]).name == "config.json")
    fitting_graph = {
        "version": FITTING_GRAPH_VERSION, "voicechat_config": voicechat_config,
        "fusion_weights": fusion_weights_from_config(json.loads(Path(voicechat_config["path"]).read_text())),
        "fusion_accumulation": "float32 before LM precision cast; same weights in prompt and audio paths",
    }
    expected3 = next(f for f in run3["artifact"]["files"] if f["path"] == "model.safetensors")
    if any(initializations["E2"][k] != expected3[k] for k in ("bytes", "sha256")):
        raise ExperimentValidationError("E2 initialization differs from the frozen Comparison 3 artifact")
    value = {"comparison": 7, "status": "prepared_not_fitted", "shared_setup": record(setup.path),
             "dataset_b": record(args.dataset_b), "dataset_b_sha256": data["manifest_sha256"],
             "sources": sources, "initializations": initializations,
             "comparison_3_run": record(args.comparison_3 / "run.json"),
             "comparison_6_run": record(args.comparison_6 / "run.json"),
             "lm_config": record(args.lm_reference / "config.json"),
             "tokenizer": record(args.lm_reference / "tokenizer.json"),
             "runtime_configuration": setup.value["candidate_runtime_configuration"],
             "system_prompts": gating.SYSTEM_PROMPTS, "arms": dataset_b.ARMS,
             "budgets": dataset_b.BUDGETS, "E1_required_first": True,
             "fitting_graph": fitting_graph}
    result = write_frozen(args.output / "experiment.json", value)
    LOG.info("Prepared experiment %s", result["manifest_sha256"])


def load_experiment(root: Path):
    experiment = read_frozen(root / "experiment.json")
    verify_record(experiment["dataset_b"])
    data = read_frozen(Path(experiment["dataset_b"]["path"]))
    dataset_b.validate_manifest(data)
    return experiment, data


def validate_fitting_graph(experiment):
    """Never resume or gate a legacy fit silently under the corrected graph."""
    graph = experiment.get("fitting_graph", {})
    if graph.get("version") != FITTING_GRAPH_VERSION:
        raise ExperimentValidationError(
            "legacy fitting graph omitted VoiceChat's function-channel weight; "
            "prepare a new experiment and refit E1, preserving the old artifacts")
    configuration = graph["voicechat_config"]
    verify_record(configuration)
    expected_path = Path(experiment["sources"]["E1"]["path"]).parent / "config.json"
    if Path(configuration["path"]).resolve() != expected_path.resolve():
        raise ExperimentValidationError("fusion configuration is not the original VoiceChat checkpoint's")
    if graph["fusion_weights"] != fusion_weights_from_config(json.loads(expected_path.read_text())):
        raise ExperimentValidationError("frozen fitting fusion weights differ from the checkpoint")
    verify_record(experiment["lm_config"])
    return graph


def validate_resume_checkpoint(saved, provenance_sha256):
    if (saved.get("fitting_graph_version") != FITTING_GRAPH_VERSION
            or saved.get("provenance_sha256") != provenance_sha256):
        raise ExperimentValidationError("optimizer checkpoint belongs to another fitting graph or candidate")


def arm_weights(experiment, arm):
    source = experiment["sources"][arm]
    verify_record(source)
    if arm == "E1":
        state = weights.load_voicechat_safetensors(Path(source["path"]))
    else:
        state = weights.load_asr(Path(source["path"]).parent, mmproj_precision=False)
    original = weights.load_voicechat_safetensors(Path(experiment["sources"]["E1"]["path"]))
    init = experiment["initializations"][arm]
    verify_record(init)
    if arm == "E2":
        mapped = weights.load_asr(Path(init["path"]).parent, mmproj_precision=False)
        final_map.assert_encoder_byte_identical(state, mapped, label="Comparison 3 initialization")
        state["proj.weight"], state["proj.bias"] = mapped["proj.weight"], mapped["proj.bias"]
    else:
        state["proj.weight"], state["proj.bias"] = original["proj.weight"], original["proj.bias"]
    state["featurizer.fb"], state["featurizer.window"] = original["featurizer.fb"], original["featurizer.window"]
    # Every candidate, including E1, has PT_ML's exact encoder configuration.
    pt_ml_config = json.loads((Path(experiment["sources"]["E2"]["path"]).parent / "config.json").read_text())
    state.config = copy.deepcopy(pt_ml_config["encoder_config"])
    return state


def cache_encoders(args):
    from safetensors.torch import save_file
    experiment, data = load_experiment(args.output)
    dataset_b.verify_audio(data)
    for arm in args.arm or dataset_b.ARMS:
        target = args.output / "activations" / arm
        state = arm_weights(experiment, arm)
        digest = final_map.encoder_byte_digest(state)
        header = {"arm": arm, "encoder": digest, "source": experiment["sources"][arm],
                  "dataset_b_sha256": data["manifest_sha256"], "runtime_configuration": experiment["runtime_configuration"],
                  "activation_precision": "F32", "inference_precision": "F32; TF32 disabled"}
        write_frozen(target / "provenance.json", header)
        model = encoder.build(state).to(args.device)
        saved = []
        rows = [r for split in data["splits"].values() for r in split]
        for index, row in enumerate(rows):
            key = row["clip_id"].replace("/", "-")
            path = target / f"{key}.safetensors"
            sidecar = target / f"{key}.json"
            expected = {"clip_id": row["clip_id"], "audio_sha256": row["sha256"],
                        "provenance_sha256": stable_json_sha256(header)}
            if sidecar.exists():
                entry = read_frozen(sidecar)
                if any(entry[k] != v for k, v in expected.items()):
                    raise ExperimentValidationError(f"cached activation provenance changed: {key}")
                verify_record(entry["activation"])
            else:
                if path.exists():
                    raise ExperimentValidationError(f"activation without provenance: {path}")
                import soundfile
                samples, rate = soundfile.read(manifests.resolve_take(Path(data["root"]), row), dtype="float32")
                if rate != features.SAMPLE_RATE:
                    raise ExperimentValidationError(f"{row['clip_id']}: {rate} Hz, not the frozen featurizer's rate")
                # log_mel computes the STFT in float64 for runtime parity; the
                # graph itself is F32, so every caller lands it back there.
                mel = features.log_mel(torch.from_numpy(samples), state["featurizer.fb"],
                                       state["featurizer.window"]).float()
                with torch.no_grad():
                    hidden = model(mel[None].to(args.device))[0].float().cpu().contiguous()
                if not torch.isfinite(hidden).all():
                    raise ExperimentValidationError(f"non-finite cached encoder output: {key}")
                target.mkdir(parents=True, exist_ok=True)
                save_file({"hidden": hidden}, path)
                entry = write_frozen(sidecar, {**expected, "activation": record(path), "shape": list(hidden.shape)})
            saved.append(entry)
            if index % 100 == 0:
                LOG.info("%s encoder clips %d/%d", arm, index + 1, len(rows))
        write_frozen(target / "index.json", {**header, "clips": saved})
        del model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def duplex_cache_geometry(root: Path, tail_frames: int) -> dict:
    """How much silence the cache has to carry, and who needs it.

    B1's timeline runs to whatever the runtime streamed; B2's runs to the end
    of its own reply.  Sizing the tail by the larger of the two is the whole
    of it -- there is no per-clip trimming here, because silence embeddings do
    not settle to a shared steady state (still ~0.6 away at k=128, and cosine
    0.85-0.995 between clips), so a shared tail would be a different tail.
    """
    b1 = read_frozen(root / "targets" / "B1-duplex" / "provenance.json")
    longest, counted = 0, 0
    for path in sorted((root / "targets" / "B2").glob("*-*.json")):
        reply = read_frozen(path).get("reply_token_ids")
        if reply:
            # BOS, the reply, EOS, and the ten pad frames that train the return
            # to listening.
            longest = max(longest, len(reply) + 12)
            counted += 1
    needed = max(int(b1["tail_frames"]), longest)
    if tail_frames < needed:
        raise ExperimentValidationError(
            f"a {tail_frames}-frame silence tail is short of the {needed} frames "
            f"B1 ({b1['tail_frames']}) and B2 ({longest}) timelines reach")
    return {"lead_frames": int(b1["lead_frames"]), "tail_frames": int(tail_frames),
            "b1_tail_frames": int(b1["tail_frames"]), "b2_longest_reply_frames": longest,
            "b2_targets_measured": counted,
            "b1_duplex_provenance_sha256": stable_json_sha256(b1)}


def cache_duplex_encoders(args):
    """Cache each clip the way the bridge streams it: silence, command, silence.

    The v1 cache holds the command alone and the timelines marked every frame
    past it -1, which `duplex_inputs` turned into an exact zero embedding.
    That is what `vc_session::run_turn` passes once the wav is spent; it is not
    what deployment does.  `bridge/server.py::_audio_loop` advances the model
    on silence, so the model hears ENCODED silence, and a projection fitted
    against the zero waits for a cue that never arrives -- the v1 pilot
    answered 22 of 24 on a zero tail and 0 of 24 on the encoded one.

    The tensors are large and go to `--activation-root`; the provenance and
    the index stay beside the experiment so the fit still verifies every byte.
    """
    from safetensors.torch import save_file

    experiment, data = load_experiment(args.output)
    dataset_b.verify_audio(data)
    geometry = duplex_cache_geometry(args.output, args.tail_frames)
    lead, tail = geometry["lead_frames"], geometry["tail_frames"]
    block = gating.DuplexPerceptionEngine.FRAME_SAMPLES
    LOG.info("duplex cache: %d lead + command + %d tail frames (B1 %d, B2 %d)",
             lead, tail, geometry["b1_tail_frames"], geometry["b2_longest_reply_frames"])
    for arm in args.arm or dataset_b.ARMS:
        index_root = args.output / "activations-duplex" / arm
        tensor_root = args.activation_root / arm
        state = arm_weights(experiment, arm)
        digest = final_map.encoder_byte_digest(state)
        header = {"arm": arm, "encoder": digest, "source": experiment["sources"][arm],
                  "dataset_b_sha256": data["manifest_sha256"],
                  "runtime_configuration": experiment["runtime_configuration"],
                  "activation_precision": "F32", "inference_precision": "F32; TF32 disabled",
                  "padding": "lead silence, command zero padded to a whole 80ms block, tail silence",
                  # The cache keeps every frame the encoder produced; which of
                  # them a streamed frame lines up with is the timeline's
                  # choice at fit time, not a property of these bytes.
                  "tensor_root": str(args.activation_root.resolve()), **geometry}
        write_frozen(index_root / "provenance.json", header)
        model = encoder.build(state).to(args.device)
        saved = []
        rows = [r for split in data["splits"].values() for r in split]
        for position, row in enumerate(rows):
            key = row["clip_id"].replace("/", "-")
            path = tensor_root / f"{key}.safetensors"
            sidecar = index_root / f"{key}.json"
            expected = {"clip_id": row["clip_id"], "audio_sha256": row["sha256"],
                        "provenance_sha256": stable_json_sha256(header)}
            if sidecar.exists():
                entry = read_frozen(sidecar)
                if any(entry[k] != v for k, v in expected.items()):
                    raise ExperimentValidationError(f"cached duplex activation provenance changed: {key}")
                verify_record(entry["activation"])
            else:
                if path.exists():
                    raise ExperimentValidationError(f"duplex activation without provenance: {path}")
                import soundfile
                samples, rate = soundfile.read(manifests.resolve_take(Path(data["root"]), row), dtype="float32")
                if rate != features.SAMPLE_RATE:
                    raise ExperimentValidationError(f"{row['clip_id']}: {rate} Hz, not the frozen featurizer's rate")
                blocks = -(-len(samples) // block)
                padded = np.zeros((lead + blocks + tail) * block, dtype="float32")
                padded[lead * block:lead * block + len(samples)] = samples
                mel = features.log_mel(torch.from_numpy(padded), state["featurizer.fb"],
                                       state["featurizer.window"]).float()
                with torch.no_grad():
                    hidden = model(mel[None].to(args.device))[0].float().cpu().contiguous()
                if not torch.isfinite(hidden).all():
                    raise ExperimentValidationError(f"non-finite cached duplex encoder output: {key}")
                if hidden.shape[0] != features.frames_out(padded.size):
                    raise ExperimentValidationError(f"duplex cache frame count moved: {key}")
                tensor_root.mkdir(parents=True, exist_ok=True)
                index_root.mkdir(parents=True, exist_ok=True)
                save_file({"hidden": hidden}, path)
                entry = write_frozen(sidecar, {**expected, "activation": record(path),
                                               "shape": list(hidden.shape), "command_blocks": blocks})
            saved.append(entry)
            if position % 100 == 0:
                LOG.info("%s duplex encoder clips %d/%d", arm, position + 1, len(rows))
        write_frozen(index_root / "index.json", {**header, "clips": saved})
        del model, state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def tokenizer(experiment):
    from tokenizers import Tokenizer
    verify_record(experiment["tokenizer"])
    return Tokenizer.from_file(experiment["tokenizer"]["path"])


def target_text(args):
    experiment, data = load_experiment(args.output)
    tok = tokenizer(experiment)
    engine = gating.TextPathEngine(args.endpoint)
    lmfile = Path(args.teacher_model)
    verify_record(experiment["lm_config"])
    header = {"path": "text_chat_native_transcript", "model": record(lmfile),
              "fitting_precision": None, "teacher_precision": "Q8_0",
              "server": engine.properties(), "binary": record(args.teacher_binary),
              "dataset_b_sha256": data["manifest_sha256"], "system_prompts": gating.SYSTEM_PROMPTS,
              "framing": gating.TEXT_FRAMING, "max_tokens": engine.max_tokens}
    output = args.output / "targets" / "B2"
    write_frozen(output / "provenance.json", header)
    identifier, quality = gating.fit_language_identifier(args.massive)
    write_frozen(output / "language_identifier.json", quality)
    rows = [r for group in data["splits"].values() for r in group if r["pool"] == "B2"]
    if args.limit:
        rows = rows[:args.limit]
    for index, row in enumerate(rows):
        for condition, system in gating.SYSTEM_PROMPTS.items():
            path = output / f"{row['clip_id'].replace('/', '-')}-{condition}.json"
            if path.exists():
                old = read_frozen(path)
                if old["teacher_provenance_sha256"] != stable_json_sha256(header):
                    raise ExperimentValidationError("target teacher provenance changed")
                continue
            rendered = gating.render_prompt("text", system, row["transcript"])
            reply = engine.generate(rendered)
            score_row = {"path": "text", "prompt_id": condition, "language": row["language"],
                         "id": row["utterance_id"], "intent": row["intent"], "utterance": row["transcript"], "reply": reply["text"]}
            scored = gating.score_rows([score_row], identifier)["rows"][0]
            expected = "en" if condition == "A_english_only" else row["language"]
            usable = scored["usable"] and scored["identified_language"] == expected and not reply.get("errors")
            write_frozen(path, {"clip_id": row["clip_id"], "condition": condition,
                               "teacher_provenance_sha256": stable_json_sha256(header),
                               "native_transcript": row["transcript"], "slot_method": row["slot_method"],
                               "reply": reply, "score": scored, "usable": bool(usable),
                               "reply_token_ids": tok.encode(scored["reply"], add_special_tokens=False).ids})
        if index % 10 == 0:
            LOG.info("B2 targets %d/%d clips under both prompts", index + 1, len(rows))


def target_audio(args):
    from lm_gating_check import _container_provenance, _docker_put
    from asr_align.interface_fit import parse_audio_trace
    import io
    import soundfile

    experiment, data = load_experiment(args.output)
    dataset_b.verify_audio(data)
    tok = tokenizer(experiment)
    header = {"path": "original_voicechat_audio", "teacher_precision": "Q8_0",
              "runtime": _container_provenance(args.container, args.teacher_model, args.teacher_mmproj),
              "dataset_b_sha256": data["manifest_sha256"], "system_prompts": gating.SYSTEM_PROMPTS,
              "n_gpu_layers": args.n_gpu_layers, "extra_decoding_seconds": 20.0,
              "frame_trace": "VC_DUMP=1; original text and function tokens at every 80ms frame"}
    output = args.output / "targets" / "B1"
    write_frozen(output / "provenance.json", header)
    identifier, quality = gating.fit_language_identifier(args.massive)
    write_frozen(output / "language_identifier.json", quality)
    trace = output / "runtime-trace.log"
    engine = gating.PerceptionPathEngine(container=args.container, model=args.teacher_model,
                                         mmproj=args.teacher_mmproj, silence="/tmp/comparison-7-b1.wav",
                                         n_gpu_layers=args.n_gpu_layers, extra_decoding_seconds=20,
                                         trace_path=trace)
    try:
        rows = [r for group in data["splits"].values() for r in group if r["pool"] == "B1"]
        if args.limit:
            rows = rows[:args.limit]
        for index, row in enumerate(rows):
            samples, rate = soundfile.read(manifests.resolve_take(Path(data["root"]), row), dtype="float32")
            wav = io.BytesIO()
            soundfile.write(wav, samples, rate, format="WAV", subtype="PCM_16")
            _docker_put(args.container, wav.getvalue(), "/tmp/comparison-7-b1.wav")
            for condition, system in gating.SYSTEM_PROMPTS.items():
                path = output / f"{row['clip_id'].replace('/', '-')}-{condition}.json"
                if path.exists():
                    old = read_frozen(path)
                    if old["teacher_provenance_sha256"] != stable_json_sha256(header):
                        raise ExperimentValidationError("B1 target teacher provenance changed")
                    continue
                offset = trace.stat().st_size
                reply = engine.generate(system, audio="/tmp/comparison-7-b1.wav")
                with trace.open("rb") as stream:
                    stream.seek(offset)
                    traced = stream.read().decode("utf-8", errors="replace")
                prefix_frames = len(tok.encode(system, add_special_tokens=False).ids) + 2
                audio_frames = features.frames_out(row["n_samples"])
                rejected, timeline = None, None
                try:
                    timeline = parse_audio_trace(traced, prefix_frames=prefix_frames, audio_frames=audio_frames)
                except ExperimentValidationError as exc:
                    rejected = str(exc)
                score_row = {"path": "perception", "prompt_id": condition, "language": "en",
                             "id": row["utterance_id"], "intent": row["intent"], "utterance": row["transcript"], "reply": reply["text"]}
                scored = gating.score_rows([score_row], identifier)["rows"][0]
                usable = scored["usable"] and scored["identified_language"] == "en" and not reply.get("errors") and not reply.get("failed") and timeline is not None
                write_frozen(path, {"clip_id": row["clip_id"], "condition": condition,
                                   "teacher_provenance_sha256": stable_json_sha256(header),
                                   "reply": reply, "score": scored, "usable": bool(usable),
                                   "timeline": timeline, "trace_rejection": rejected,
                                   "runtime_trace": traced})
            if index % 10 == 0:
                LOG.info("B1 targets %d/%d clips under both prompts", index + 1, len(rows))
    finally:
        engine.close()


def target_audio_duplex(args):
    """B1, regenerated on the turn path deployment actually uses.

    `target_audio` drives `vc_session::run_turn`, which forces the opening BOS
    and hands the model an exact zero audio embedding once the wav is spent.
    Neither happens in `duplex_step`, which is what the Realtime bridge runs, so
    a projection fitted against those traces learns to wait for a cue that never
    arrives.  This streams the command 80 ms at a time and then keeps the clock
    running on silence, exactly as `bridge/server.py::_audio_loop` does.

    Barge-in is not supervised: it is FT_EN's own behaviour, it stays in the
    frozen LM, and teaching it here would train the student to answer before it
    has heard the request.
    """

    from lm_gating_check import _container_provenance
    import io
    import soundfile

    experiment, data = load_experiment(args.output)
    dataset_b.verify_audio(data)
    header = {"path": "original_voicechat_audio_duplex", "teacher_precision": "Q8_0",
              "runtime": _container_provenance(args.container, args.teacher_model, args.teacher_mmproj),
              "dataset_b_sha256": data["manifest_sha256"], "system_prompts": gating.SYSTEM_PROMPTS,
              "n_gpu_layers": args.n_gpu_layers, "lead_frames": args.lead_frames,
              "tail_frames": args.tail_frames,
              "onset_tolerance_frames": interface_fit.DUPLEX_ONSET_TOLERANCE_FRAMES,
              "max_onset_frames": interface_fit.DUPLEX_MAX_ONSET_FRAMES,
              "turn_path": "vc_session::duplex_step via duplex_start/audio_frame",
              "command_boundary": "summed encoder frames acknowledged by duplex_frame",
              "rules": ["the model opens its own turn: duplex_step clears want_bos and hold_bos",
                        "reject a turn opened more than the tolerance before the command ended",
                        "after the command the model hears encoded silence, never a zero embedding"],
              "frame_trace": "VC_DUMP=1; original text and function tokens at every 80ms frame",
              "trace_slicing": ("the turn is the last contiguous run from the system prompt's last "
                                "frame; a byte-offset slice can open with the previous turn's "
                                "block-buffered stderr")}
    output = args.output / "targets" / "B1-duplex"
    write_frozen(output / "provenance.json", header)
    identifier, quality = gating.fit_language_identifier(args.massive)
    write_frozen(output / "language_identifier.json", quality)
    trace = output / "runtime-trace.log"

    engine = gating.DuplexPerceptionEngine(
        container=args.container, model=args.teacher_model, mmproj=args.teacher_mmproj,
        n_gpu_layers=args.n_gpu_layers, extra_decoding_seconds=args.extra_decoding_seconds,
        session_seconds=args.session_seconds, trace_path=trace)
    try:
        rows = [r for group in data["splits"].values() for r in group if r["pool"] == "B1"]
        if args.limit:
            rows = rows[:args.limit]
        LOG.info("B1 duplex regeneration: %d clips under %d prompts", len(rows), len(gating.SYSTEM_PROMPTS))
        for index, row in enumerate(rows):
            samples, rate = soundfile.read(manifests.resolve_take(Path(data["root"]), row), dtype="float32")
            if rate != features.SAMPLE_RATE:
                raise ExperimentValidationError(f"{row['clip_id']}: {rate} Hz, not the featurizer's rate")
            for condition, system in gating.SYSTEM_PROMPTS.items():
                path = output / f"{row['clip_id'].replace('/', '-')}-{condition}.json"
                if path.exists():
                    old = read_frozen(path)
                    if old["teacher_provenance_sha256"] != stable_json_sha256(header):
                        raise ExperimentValidationError("B1 duplex target teacher provenance changed")
                    continue
                offset = trace.stat().st_size
                reply = engine.generate(system, samples, lead_frames=args.lead_frames,
                                        tail_frames=args.tail_frames)
                with trace.open("rb") as stream:
                    stream.seek(offset)
                    traced = stream.read().decode("utf-8", errors="replace")
                # The runtime reports where conditioning ended; a tokenizer
                # estimate of it is off by a frame on some prompts.
                prefix_frames = reply["system_end_t"]
                timeline, rejected = None, None
                try:
                    timeline = parse_duplex_trace(traced, prefix_frames=prefix_frames)
                except ExperimentValidationError as exc:
                    rejected = str(exc)
                score_row = {"path": "perception", "prompt_id": condition, "language": "en",
                             "id": row["utterance_id"], "intent": row["intent"],
                             "utterance": row["transcript"], "reply": reply["text"]}
                scored = gating.score_rows([score_row], identifier)["rows"][0]
                rejections = duplex_turn_rejections(opened=reply["opened"],
                                                    onset=reply["onset_frames_past_command"],
                                                    spoken=reply["spoken"])
                if reply.get("errors"):
                    rejections.append("runtime error")
                if not scored["usable"]:
                    rejections.append("reply not usable")
                if scored["identified_language"] != "en":
                    rejections.append(f"reply identified as {scored['identified_language']}")
                if timeline is None:
                    rejections.append(f"trace rejected: {rejected}")
                write_frozen(path, {"clip_id": row["clip_id"], "condition": condition,
                                    "teacher_provenance_sha256": stable_json_sha256(header),
                                    "reply": reply, "score": scored,
                                    "usable": not rejections, "rejections": rejections,
                                    "command_encoder_frames": reply["command_encoder_frames"],
                                    "lead_frames": reply["lead_frames"],
                                    "onset_frames_past_command": reply["onset_frames_past_command"],
                                    "timeline": timeline, "trace_rejection": rejected,
                                    "runtime_trace": traced})
            if index % 10 == 0:
                LOG.info("B1 duplex targets %d/%d clips under both prompts", index + 1, len(rows))
    finally:
        engine.close()


def target_directory(args, pool: str) -> str:
    """Which recorded pool a run supervises B1 from.

    `B1` is the v1 whole-wav trace: forced BOS, barge-in suppressed, a zero
    embedding past the command.  `B1-duplex` is the same clips regenerated on
    the turn path deployment actually runs.
    """
    return args.b1_directory if pool == "B1" else pool


def freeze_targets(args):
    experiment, data = load_experiment(args.output)
    entries, summary = [], {}
    for split, rows in data["splits"].items():
        for row in rows:
            pair = []
            for condition in gating.SYSTEM_PROMPTS:
                path = (args.output / "targets" / target_directory(args, row["pool"])
                        / f"{row['clip_id'].replace('/', '-')}-{condition}.json")
                if not path.exists():
                    raise ExperimentValidationError(f"missing teacher target: {path}")
                target = read_frozen(path)
                pair.append((condition, target, record(path)))
            retained = all(t["usable"] for _, t, _ in pair)
            for condition, target, file in pair:
                cell = f"{split}/{row['pool']}/{row['language']}/{condition}"
                counts = summary.setdefault(cell, {"offered": 0, "usable": 0, "paired_retained": 0})
                counts["offered"] += 1
                counts["usable"] += int(target["usable"])
                counts["paired_retained"] += int(retained)
                entries.append({"clip_id": row["clip_id"], "condition": condition, "split": split,
                                "retained": retained, "target": file})
    if any(v["paired_retained"] == 0 for v in summary.values()):
        raise ExperimentValidationError("teacher filtering emptied a Dataset B language/prompt cell")
    write_frozen(args.output / "targets" / args.targets_index, {
        "teacher_paths": data["teacher_paths"], "dataset_b_sha256": data["manifest_sha256"],
        "retention_rule": "retain a clip only if both prompt conditions pass the frozen gate",
        "b1_directory": args.b1_directory, "teacher_quality": summary, "entries": entries,
        "provenance": {pool: record(args.output / "targets" / target_directory(args, pool) / "provenance.json")
                       for pool in ("B1", "B2")}})


def load_training_items(root, data, targets, budget, arm, *, split="train",
                        frame_offset=interface_fit.CACHE_FRAME_OFFSET):
    from safetensors.torch import load_file
    allowed = set(dataset_b.budget_clip_ids(data, budget)) if split == "train" else {r["clip_id"] for r in data["splits"][split]}
    rows = {r["clip_id"]: r for r in data["splits"][split]}
    cache = read_frozen(root / "activations-duplex" / arm / "index.json")
    if cache["dataset_b_sha256"] != data["manifest_sha256"]:
        raise ExperimentValidationError("encoder cache uses another training manifest")
    lead = int(cache["lead_frames"])
    cached = {r["clip_id"]: r for r in cache["clips"]}
    items = []
    for entry in targets["entries"]:
        key = entry["clip_id"]
        if entry["split"] != split or key not in allowed or not entry["retained"]:
            continue
        verify_record(entry["target"])
        target = read_frozen(Path(entry["target"]["path"]))
        activation = cached[key]
        verify_record(activation["activation"])
        hidden = load_file(activation["activation"]["path"])["hidden"]
        blocks = int(activation["command_blocks"])
        if rows[key]["pool"] == "B1":
            # The trace begins at the bridge's first frame, so it covers the
            # lead silence too and indexes the cache from its own start.
            if int(target["lead_frames"]) != lead:
                raise ExperimentValidationError(f"{key}: teacher and cache disagree on the lead silence")
            # The runtime acknowledged one encoder frame per streamed block; if
            # it did not, the cache is not the audio the teacher heard.
            if int(target["command_encoder_frames"]) - lead != blocks:
                raise ExperimentValidationError(
                    f"{key}: the teacher consumed {int(target['command_encoder_frames']) - lead} command "
                    f"frames where the cache holds {blocks} blocks")
            timeline = dict(target["timeline"])
            command_frames = int(target["command_encoder_frames"])
            timeline["audio_indices"] = interface_fit.duplex_audio_indices(
                len(timeline["text_tokens"]), offset=frame_offset)
        else:
            timeline = text_target_timeline(target["reply_token_ids"], blocks,
                                            lead_frames=lead, offset=frame_offset)
            command_frames = blocks
        items.append({"clip_id": key, "condition": entry["condition"], "pool": rows[key]["pool"],
                      "language": rows[key]["language"], "features": hidden, "timeline": timeline,
                      "command_frames": command_frames})
    return items, cache


def fit(args):
    from safetensors.torch import load_file, save_file
    experiment, data = load_experiment(args.output)
    fitting_graph = validate_fitting_graph(experiment)
    targets = read_frozen(args.output / "targets" / args.targets_index)
    if targets["dataset_b_sha256"] != data["manifest_sha256"]:
        raise ExperimentValidationError("teacher targets use another Dataset B")
    if args.arm != "E1":
        gate_path = args.output / "E1_english_gate.json"
        gate = read_frozen(gate_path)
        if not gate.get("passed") or gate.get("experiment_sha256") != experiment["manifest_sha256"]:
            raise ExperimentValidationError("E1 must pass the English deployment control before fitting other arms")
    output = args.output / "fits" / f"{args.arm}-{args.budget}-{args.precision}"
    if (output / "result.json").exists():
        raise ExperimentValidationError("fit is already frozen; choose a new experiment for changed settings")
    items, cache = load_training_items(args.output, data, targets, args.budget, args.arm,
                                       frame_offset=args.frame_offset)
    validation, _ = load_training_items(args.output, data, targets, args.budget, args.arm,
                                        split="validation", frame_offset=args.frame_offset)
    if not items or not validation:
        raise ExperimentValidationError("empty training or validation selection")
    LOG.info("%s %d%% %s: %d training and %d validation clip/prompt examples; verifying source checkpoint",
             args.arm, args.budget, args.precision, len(items), len(validation))
    source = arm_weights(experiment, args.arm)
    if final_map.encoder_byte_digest(source) != cache["encoder"]:
        raise ExperimentValidationError("encoder cache differs from the arm source")
    projection = torch.nn.Linear(source["proj.weight"].shape[1], source["proj.weight"].shape[0]).to(args.device)
    with torch.no_grad():
        projection.weight.copy_(source["proj.weight"])
        projection.bias.copy_(source["proj.bias"])
    del source
    gc.collect()
    LOG.info("Source verified; loading the frozen %s LM with fusion weights %s",
             args.precision, fitting_graph["fusion_weights"])
    lm = FrozenVoiceChatLM(Path(experiment["sources"]["E1"]["path"]), Path(experiment["lm_config"]["path"]),
                           precision=args.precision, device=args.device)
    tok = tokenizer(experiment)
    prefixes = {condition: lm.cache_prompt([1, *tok.encode(prompt, add_special_tokens=False).ids, 2])
                for condition, prompt in gating.SYSTEM_PROMPTS.items()}
    calibration_path = args.output / "loss_calibration.json"
    if calibration_path.exists():
        calibration = read_frozen(calibration_path)
        if (calibration.get("fitting_graph_sha256") != stable_json_sha256(fitting_graph)
                or calibration.get("dataset_b_sha256") != data["manifest_sha256"]):
            raise ExperimentValidationError("loss calibration belongs to another fitting graph or dataset")
    else:
        if args.arm != "E1":
            raise ExperimentValidationError("E1 gradient calibration must be frozen first")
        LOG.info("Calibrating E1 pool weights under fitting graph %s", FITTING_GRAPH_VERSION)
        norms, calibration_ids = {"B1": [], "B2": []}, []
        # Equal conditions/languages in calibration; no validation/test scores.
        cells = sorted({(i["pool"], i["language"], i["condition"]) for i in items})
        for cell in cells:
            selected = sorted([i for i in items if (i["pool"], i["language"], i["condition"]) == cell],
                              key=lambda i: stable_json_sha256([0, i["clip_id"]]))[:4]
            for item in selected:
                projection.zero_grad(set_to_none=True)
                loss = token_loss(projection, lm, item["features"].to(args.device), item["timeline"], prefixes[item["condition"]])
                loss.backward()
                norm = torch.sqrt(sum(p.grad.float().square().sum() for p in projection.parameters()))
                norms[item["pool"]].append(float(norm))
                calibration_ids.append([item["clip_id"], item["condition"]])
        calibration = write_frozen(calibration_path, {**calibrate_pool_weights(norms), "items": calibration_ids,
                                                       "precision": args.precision, "dataset_b_sha256": data["manifest_sha256"],
                                                       "fitting_graph_sha256": stable_json_sha256(fitting_graph)})
    LOG.info("Frozen loss weights: %s", calibration["weights"])
    provenance = candidate_provenance(arm=args.arm, budget=args.budget, manifest=data,
                                     source=experiment["sources"][args.arm], initialization=experiment["initializations"][args.arm],
                                     language_model={**experiment["sources"]["E1"], "fitting_precision": args.precision},
                                     targets=targets, calibration=calibration, fitting_graph=fitting_graph)
    settings = {"epochs": args.epochs, "learning_rate": args.learning_rate, "gradient_accumulation": args.accumulate,
                "optimizer": "AdamW", "weight_decay": 0.0, "seed": args.seed, "gradient_clip_norm": 1.0}
    write_frozen(output / "provenance.json", {**provenance, "settings": settings})
    optimizer = torch.optim.AdamW(projection.parameters(), lr=args.learning_rate, weight_decay=0.0)
    step, start_epoch, cursor = 0, 0, 0
    checkpoint_path = output / "resume.pt"
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        validate_resume_checkpoint(saved, provenance["provenance_sha256"])
        projection.load_state_dict(saved["projection"])
        optimizer.load_state_dict(saved["optimizer"])
        step, start_epoch, cursor = saved["step"], saved["epoch"], saved["cursor"]
        LOG.info("Resuming %s at epoch=%d examples=%d/%d step=%d",
                 output.name, start_epoch + 1, cursor, len(items), step)
    for epoch in range(start_epoch, args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(items), generator=generator).tolist()
        begin = cursor if epoch == start_epoch else 0
        for start in range(begin, len(order), args.accumulate):
            batch = order[start:start + args.accumulate]
            optimizer.zero_grad(set_to_none=True)
            measured = 0.0
            for index in batch:
                item = items[index]
                loss = token_loss(projection, lm, item["features"].to(args.device), item["timeline"], prefixes[item["condition"]])
                loss = loss * calibration["weights"][item["pool"]] / len(batch)
                if not torch.isfinite(loss):
                    raise ExperimentValidationError("non-finite projection training loss")
                loss.backward()
                measured += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            lm.assert_frozen()
            step += 1
            LOG.info("%s %d%% %s epoch=%d examples=%d/%d step=%d loss=%.5f grad=%.5f",
                     args.arm, args.budget, args.precision, epoch + 1, start + len(batch), len(order), step, measured, norm)
            temp = output / "resume.tmp"
            torch.save({"projection": projection.state_dict(), "optimizer": optimizer.state_dict(),
                        "fitting_graph_version": FITTING_GRAPH_VERSION,
                        "provenance_sha256": provenance["provenance_sha256"],
                        "step": step, "epoch": epoch, "cursor": start + len(batch)}, temp)
            temp.replace(checkpoint_path)
    LOG.info("%s optimizer updates complete; validating %d clip/prompt examples before writing result.json",
             output.name, len(validation))
    cells = scored_cells(projection, lm, validation, prefixes, args.device)
    path = output / "projection.safetensors"
    save_file({"proj.weight": projection.weight.detach().cpu().contiguous(),
               "proj.bias": projection.bias.detach().cpu().contiguous()}, path)
    write_frozen(output / "result.json", {"status": "fit_complete_evaluations_pending", "projection": record(path),
                                          "provenance_sha256": provenance["provenance_sha256"], "steps": step,
                                          "encoder": cache["encoder"], "train_examples": len(items),
                                          "validation_token_ce": cells})
    LOG.info("Fit complete: %s (%d optimizer steps); English gate and deployment evaluations remain separate",
             output / "result.json", step)


def linear_from(weight: torch.Tensor, bias: torch.Tensor, device: str) -> torch.nn.Linear:
    projection = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    with torch.no_grad():
        projection.weight.copy_(weight)
        projection.bias.copy_(bias)
    return projection


def read_fit(root: Path, arm: str, budget: int, precision: str):
    """Load one frozen fit and the projection it actually wrote."""

    fit_root = root / "fits" / f"{arm}-{budget}-{precision}"
    result = read_frozen(fit_root / "result.json")
    verify_record(result["projection"])
    return fit_root, result, read_frozen(fit_root / "provenance.json")


def scored_cells(projection, lm, items, prefixes, device) -> dict:
    cells: dict[str, list[float]] = {}
    with torch.no_grad():
        for index, item in enumerate(items, 1):
            loss = token_loss(projection, lm, item["features"].to(device), item["timeline"],
                              prefixes[item["condition"]])
            cells.setdefault(f"{item['pool']}/{item['language']}/{item['condition']}", []).append(float(loss))
            if index % 100 == 0 or index == len(items):
                LOG.info("Validation scored %d/%d clip/prompt examples", index, len(items))
    return {key: {"n": len(values), "mean": sum(values) / len(values)} for key, values in cells.items()}


def duplex_cells(projection, lm, items, prefixes, device) -> dict:
    """Free-running turn taking, one cell per prompt condition.

    `scored_cells` measures teacher-forced cross-entropy, which the v1
    projection passed while answering nothing in deployment.  This measures
    whether the projection still decides to speak when nothing is fed back to
    it, under the three conditions the Realtime bridge imposes.
    """
    cells: dict[str, dict] = {}
    for item in items:
        result = interface_fit.duplex_free_run(
            projection, lm, item["features"].to(device), item["timeline"],
            prefixes[item["condition"]], command_frames=item["command_frames"])
        cell = cells.setdefault(item["condition"], {"n": 0, "opened": 0, "onsets": [], "silent": []})
        cell["n"] += 1
        cell["opened"] += int(result["opened"])
        if result["opened"]:
            cell["onsets"].append(int(result["onset_frames_past_command"]))
        else:
            cell["silent"].append(item["clip_id"])
    return cells


def english_gate(args):
    """The control every other arm waits on: did fitting break the loop?"""

    from safetensors.torch import load_file

    experiment, data = load_experiment(args.output)
    validate_fitting_graph(experiment)
    targets = read_frozen(args.output / "targets" / args.targets_index)
    if targets["dataset_b_sha256"] != data["manifest_sha256"]:
        raise ExperimentValidationError("teacher targets use another Dataset B")
    fit_root, result, _ = read_fit(args.output, "E1", args.budget, args.precision)
    validation, cache = load_training_items(args.output, data, targets, args.budget, "E1",
                                           split="validation", frame_offset=args.frame_offset)
    english = [item for item in validation if item["pool"] == "B1"]
    if not english:
        raise ExperimentValidationError("the English gate needs held-out B1 clips")
    LOG.info("E1 English gate: verifying the source and loading %s LM; %d clips under both prompts",
             args.precision, len(english) // len(gating.SYSTEM_PROMPTS))
    source = arm_weights(experiment, "E1")
    if final_map.encoder_byte_digest(source) != cache["encoder"] or cache["encoder"] != result["encoder"]:
        raise ExperimentValidationError("the gate encoder differs from the fitted E1 source")
    lm = FrozenVoiceChatLM(Path(experiment["sources"]["E1"]["path"]), Path(experiment["lm_config"]["path"]),
                           precision=args.precision, device=args.device)
    tok = tokenizer(experiment)
    prefixes = {condition: lm.cache_prompt([1, *tok.encode(prompt, add_special_tokens=False).ids, 2])
                for condition, prompt in gating.SYSTEM_PROMPTS.items()}
    fitted = load_file(result["projection"]["path"])
    initial_projection = linear_from(source["proj.weight"], source["proj.bias"], args.device)
    fitted_projection = linear_from(fitted["proj.weight"], fitted["proj.bias"], args.device)
    # The initialization is E1's own untouched FT_EN projection, so this is the
    # design record's "trained proj against original proj on English" exactly.
    measured = {
        "initialization": scored_cells(initial_projection, lm, english, prefixes, args.device),
        "fitted": scored_cells(fitted_projection, lm, english, prefixes, args.device),
    }
    verdict = english_gate_verdict(measured["fitted"], measured["initialization"], tolerance=args.tolerance)
    # Teacher forcing hides the one thing the pilot got wrong, so the gate does
    # not stop at cross-entropy: it also runs the turn free, with the untouched
    # projection alongside as the control that says the harness works.
    LOG.info("E1 duplex control: free running %d held-out clip/prompt examples under both projections",
             len(english))
    duplex = {
        "initialization": duplex_cells(initial_projection, lm, english, prefixes, args.device),
        "fitted": duplex_cells(fitted_projection, lm, english, prefixes, args.device),
    }
    duplex_verdict = interface_fit.duplex_gate_verdict(
        duplex["fitted"], duplex["initialization"], tolerance=args.open_rate_tolerance)
    value = write_frozen(args.output / "E1_english_gate.json",
                         {"arm": "E1", "budget": args.budget, "precision": args.precision,
                          "experiment_sha256": experiment["manifest_sha256"],
                          "dataset_b_sha256": data["manifest_sha256"],
                          "fit_result": record(fit_root / "result.json"),
                          "projection": dict(result["projection"]), "encoder": cache["encoder"],
                          "frame_offset": args.frame_offset,
                          "measured": measured, "duplex_measured": duplex,
                          "duplex": duplex_verdict,
                          **verdict, "passed": bool(verdict["passed"] and duplex_verdict["passed"])})
    LOG.info("E1 English gate %s: %+.5f nats against the initialization over %d held-out clip/prompt examples",
             "passed" if verdict["passed"] else "FAILED", value["delta"], value["clips_scored"])
    LOG.info("E1 duplex gate %s: opened %.1f%% of turns against the initialization's %.1f%%%s",
             "passed" if duplex_verdict["passed"] else "FAILED",
             100 * duplex_verdict["fitted_open_rate"], 100 * duplex_verdict["initialization_open_rate"],
             "" if duplex_verdict["control_calibrated"] else " (CONTROL DID NOT OPEN EVERY TURN)")
    if not value["passed"]:
        raise ExperimentValidationError(
            "E1 did not match its initialization on English; the fitting loop is broken"
            if not verdict["passed"] else
            "E1 lost its ability to open a duplex turn; the fit would be silent in deployment")


def export_artifact(args):
    """Write the directory the deployment converter can consume on its own."""

    import sys

    from safetensors.torch import load_file

    experiment, _ = load_experiment(args.output)
    fit_root, result, provenance = read_fit(args.output, args.arm, args.budget, args.precision)
    source = arm_weights(experiment, args.arm)
    if final_map.encoder_byte_digest(source) != result["encoder"]:
        raise ExperimentValidationError("the fitted encoder differs from the arm source")
    fitted = load_file(result["projection"]["path"])
    if sorted(fitted) != ["proj.bias", "proj.weight"]:
        raise ExperimentValidationError("a fitted interface exports exactly proj.weight and proj.bias")
    for key in ("proj.weight", "proj.bias"):
        if fitted[key].shape != source[key].shape or not bool(torch.isfinite(fitted[key]).all()):
            raise ExperimentValidationError(f"fitted {key} changed shape or is not finite")
    artifact = args.output / "artifacts" / fit_root.name
    report = {"schema_version": "1.0", "artifact_kind": interface_fit.ARTIFACT_KIND,
              "comparison": interface_fit.COMPARISON, "candidate_id": fit_root.name,
              "arm": args.arm, "arm_definition": dataset_b.ARMS[args.arm],
              "data_budget_percent": args.budget, "fitting_precision": args.precision,
              "map": None, "lambda": None, "alpha": None,
              "projection_dim": int(fitted["proj.weight"].shape[0]),
              "source": experiment["sources"][args.arm]["path"],
              "initialization": dict(experiment["initializations"][args.arm]),
              "shared_setup": dict(experiment["shared_setup"]),
              "runtime_configuration_source": "M/PT_ML", "runtime_configuration_exact": True,
              "provenance": provenance, "fit": dict(result),
              "command": [str(Path(sys.executable).resolve()), *sys.argv]}
    # PT_ML supplies config.json and processor_config.json for every arm, which
    # is invariant 5 and not a claim about whose encoder tensors these are.
    pt_ml = Path(experiment["sources"]["E2"]["path"]).parent
    export.export(artifact, source=pt_ml,
                  encoder={key: value.detach().float().cpu().numpy()
                           for key, value in source.items() if key.startswith("encoder.")},
                  proj_weight=fitted["proj.weight"].detach().float().cpu().numpy(),
                  proj_bias=fitted["proj.bias"].detach().float().cpu().numpy(),
                  featurizer={"fb": source["featurizer.fb"].detach().float().cpu().numpy(),
                              "window": source["featurizer.window"].detach().float().cpu().numpy()},
                  report=report)
    reloaded = weights.load_asr(artifact, mmproj_precision=False)
    assert_runtime_config_inherited(reloaded.config, source.config)
    report["checks"] = {
        # Invariant 6: fitting moved proj and nothing else.
        "encoder_byte_identical_to_arm_source": final_map.assert_encoder_byte_identical(
            source, reloaded, label=f"{args.arm} encoder source"),
        "exported_interface_matches_fit": baseline.assert_exact_tensors(
            {**fitted, "featurizer.fb": source["featurizer.fb"], "featurizer.window": source["featurizer.window"]},
            reloaded, keys=baseline.ATTACHED_KEYS),
    }
    (artifact / "interface_fit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    write_frozen(args.output / "artifacts" / f"{fit_root.name}.json",
                 {"artifact": str(artifact.resolve()), "arm": args.arm, "budget": args.budget,
                  "precision": args.precision, "experiment_sha256": experiment["manifest_sha256"],
                  "provenance_sha256": provenance["provenance_sha256"],
                  "files": [record(path) for path in sorted(artifact.iterdir()) if path.is_file()]})
    LOG.info("Exported %s to %s", fit_root.name, artifact)


def evaluate_artifact(args):
    """Shared pre/post metrics using the exact frozen Comparison 1 arrays.

    These encoder/projection metrics do not consume a system prompt. The two
    output-language conditions still require separate speech-to-action tables.
    """
    import sys
    import tempfile

    import direct_task_arithmetic as direct_runner
    import pt_ml_baseline as baseline_runner
    from asr_align import data as audio_data, direct, evaluation
    from safetensors.torch import load_file

    experiment, _ = load_experiment(args.output)
    fit_root, result, provenance = read_fit(args.output, args.arm, args.budget, args.precision)
    exported = read_frozen(args.output / "artifacts" / f"{fit_root.name}.json")
    if (exported["experiment_sha256"] != experiment["manifest_sha256"]
            or exported["provenance_sha256"] != result["provenance_sha256"]):
        raise ExperimentValidationError("exported artifact belongs to another fit or experiment")
    for file in exported["files"]:
        verify_record(file)
    artifact = Path(exported["artifact"])
    output = args.output / "evaluations" / fit_root.name
    if output.exists():
        raise ExperimentValidationError("shared evaluation is already frozen; use a new output directory")
    LOG.info("Verifying frozen shared setup and Comparison 1 reference")
    verify_record(experiment["shared_setup"])
    shared = baseline.load_shared_setup(Path(experiment["shared_setup"]["path"]))
    reference = direct.load_baseline_reference(args.baseline, shared)
    work = Path(reference.run["runtime_reader"]["path"]).resolve()
    reader = baseline_runner._runtime_reader_provenance(work)
    pre = weights.load_asr(artifact, mmproj_precision=False)
    if final_map.encoder_byte_digest(pre) != result["encoder"]:
        raise ExperimentValidationError("exported encoder differs from the fitted arm")
    fitted = load_file(result["projection"]["path"])
    baseline.assert_exact_tensors(fitted, pre, keys=("proj.weight", "proj.bias"))
    pt_ml_config = json.loads((shared.pt_ml_path / "config.json").read_text())
    assert_runtime_config_inherited(pre.config, pt_ml_config["encoder_config"])
    post = weights.load_mmproj(args.deployment, work, config=pre.config)
    simulated = weights.load_asr(artifact, mmproj_precision=True)
    quantization_check = baseline.assert_exact_tensors(simulated, post)
    quantization = baseline.quantization_report(pre, post)
    del simulated
    gc.collect()
    fleurs = manifests.load_manifest(shared.fleurs_manifest)
    manifests.verify_audio_files(fleurs, root=Path(fleurs["root"]))
    clips = audio_data.from_frozen_manifest(shared.librispeech_manifest)["validation"]
    if not clips:
        raise ExperimentValidationError("empty frozen LibriSpeech validation split")
    device = torch.device(args.device)
    torch.manual_seed(shared.seed)
    torch.use_deterministic_algorithms(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    results, files = {}, []
    for stage, state in (("pre_quantization", pre), ("post_quantization", post)):
        LOG.info("Shared evaluation: %s %s", fit_root.name, stage)
        model = encoder.build(state).to(device)
        with torch.inference_mode():
            bundle = {
                "librispeech": direct_runner._collect_librispeech(
                    model, clips, batch_size=args.batch,
                    eval_frames=int(reference.run["evaluation"]["english_frame_cap"]),
                    device=device, mel_filters=state["featurizer.fb"], window=state["featurizer.window"],
                    candidate_name=fit_root.name),
                "fleurs": direct_runner._collect_fleurs(
                    model, fleurs, device=device, mel_filters=state["featurizer.fb"],
                    window=state["featurizer.window"], candidate_name=fit_root.name),
            }
        frozen = direct_runner._read_reference_bundle(reference, stage, fleurs["languages"])
        measured = direct_runner._evaluate_stage(
            candidate_name=fit_root.name, weight=None, stage=stage, candidate_bundle=bundle,
            reference_bundle=frozen, manifest_hashes=shared.manifest_hashes,
            seed=shared.seed, comparison=interface_fit.COMPARISON)
        evaluation.write_result(temporary / f"{stage}.json", measured)
        direct_runner._write_candidate_embeddings(
            temporary / f"{stage}.safetensors", bundle, candidate_name=fit_root.name,
            weight=None, stage=stage, manifest_hashes=shared.manifest_hashes,
            comparison=interface_fit.COMPARISON)
        results[stage] = measured
        del model, bundle, frozen
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    evaluation.validate_precision_pair(results["pre_quantization"], results["post_quantization"])
    for file in sorted(temporary.iterdir()):
        files.append({**record(file), "path": str((output / file.name).resolve())})
    write_frozen(temporary / "run.json", {
        "comparison": interface_fit.COMPARISON, "candidate_id": fit_root.name,
        "status": "shared_evaluation_complete_deployment_endpoints_separate",
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "fit_result": record(fit_root / "result.json"),
        "fit_provenance": record(fit_root / "provenance.json"),
        "fitting_precision": args.precision, "system_prompts": provenance["system_prompts"],
        "shared_setup": record(shared.path), "manifests": shared.manifest_hashes,
        "artifact_index": record(args.output / "artifacts" / f"{fit_root.name}.json"),
        "deployment": record(args.deployment), "runtime_reader": reader,
        "paired_baseline_run": record(reference.run_path),
        "paired_baseline_embeddings": record(reference.embeddings_path),
        "exact_frozen_baseline_arrays_used": True,
        "actual_artifact_matches_rounding_model": quantization_check,
        "quantization": quantization,
        "precision_delta": baseline.precision_metric_delta(results["pre_quantization"], results["post_quantization"]),
        "files": files,
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                        "device": str(device), "seed": shared.seed, "tf32": False,
                        "deterministic_algorithms": True},
    })
    temporary.rename(output)
    LOG.info("Shared evaluation complete: %s; speech-to-action and MASSIVE scoring remain separate", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("shared-setup", "dataset-b", "comparison-3", "comparison-6", "lm-reference"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    prep.set_defaults(run=prepare)
    cache = sub.add_parser("cache")
    cache.add_argument("--arm", choices=dataset_b.ARMS, action="append")
    cache.set_defaults(run=cache_encoders)
    duplex_cache = sub.add_parser("cache-duplex")
    duplex_cache.add_argument("--arm", choices=dataset_b.ARMS, action="append")
    duplex_cache.add_argument("--tail-frames", type=int, default=213)
    # The padded runs are an order of magnitude larger than the command-only
    # cache; /srv/fast has no room for them.
    duplex_cache.add_argument("--activation-root", type=Path, required=True)
    duplex_cache.set_defaults(run=cache_duplex_encoders)
    target = sub.add_parser("targets-text")
    target.add_argument("--endpoint", required=True)
    target.add_argument("--teacher-model", type=Path, required=True)
    target.add_argument("--teacher-binary", type=Path, required=True)
    target.add_argument("--massive", type=Path, required=True)
    target.add_argument("--limit", type=int)
    target.set_defaults(run=target_text)
    audio = sub.add_parser("targets-audio")
    audio.add_argument("--container", default="nemotron-voicechat")
    audio.add_argument("--teacher-model", default="/models/nemotron_voicechat_11b-stt-llm-Q8_0.gguf")
    audio.add_argument("--teacher-mmproj", default="/models/mmproj-voicechat-perception-Q8_0.gguf")
    audio.add_argument("--n-gpu-layers", type=int, default=24)
    audio.add_argument("--massive", type=Path, required=True)
    audio.add_argument("--limit", type=int)
    audio.set_defaults(run=target_audio)
    duplex = sub.add_parser("targets-audio-duplex")
    duplex.add_argument("--container", default="nemotron-voicechat")
    duplex.add_argument("--teacher-model", default="/models/nemotron_voicechat_11b-stt-llm-Q8_0.gguf")
    duplex.add_argument("--teacher-mmproj", default="/models/mmproj-voicechat-perception-Q8_0.gguf")
    duplex.add_argument("--n-gpu-layers", type=int, default=24)
    duplex.add_argument("--massive", type=Path, required=True)
    duplex.add_argument("--lead-frames", type=int, default=8)
    duplex.add_argument("--tail-frames", type=int, default=150)
    duplex.add_argument("--extra-decoding-seconds", type=float, default=50.0)
    duplex.add_argument("--session-seconds", type=float, default=180.0)
    duplex.add_argument("--limit", type=int)
    duplex.set_defaults(run=target_audio_duplex)
    freeze = sub.add_parser("freeze-targets")
    freeze.set_defaults(run=freeze_targets)
    freeze.add_argument("--b1-directory", choices=("B1", "B1-duplex"), default="B1-duplex")
    train = sub.add_parser("fit")
    train.add_argument("--arm", choices=dataset_b.ARMS, required=True)
    train.add_argument("--budget", type=int, choices=dataset_b.BUDGETS, required=True)
    train.add_argument("--precision", choices=("nf4", "bf16_cpu_offload"), default="nf4")
    train.add_argument("--epochs", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--accumulate", type=int, default=8)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(run=fit)
    verdict = sub.add_parser("gate")
    verdict.add_argument("--budget", type=int, choices=dataset_b.BUDGETS, default=100)
    verdict.add_argument("--precision", choices=("nf4", "bf16_cpu_offload"), default="nf4")
    verdict.add_argument("--tolerance", type=float, default=interface_fit.GATE_TOLERANCE_NATS)
    verdict.add_argument("--open-rate-tolerance", type=float,
                         default=interface_fit.DUPLEX_OPEN_RATE_TOLERANCE)
    verdict.set_defaults(run=english_gate)
    artifact = sub.add_parser("export")
    artifact.add_argument("--arm", choices=dataset_b.ARMS, required=True)
    artifact.add_argument("--budget", type=int, choices=dataset_b.BUDGETS, required=True)
    artifact.add_argument("--precision", choices=("nf4", "bf16_cpu_offload"), default="nf4")
    artifact.set_defaults(run=export_artifact)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--arm", choices=dataset_b.ARMS, required=True)
    evaluate.add_argument("--budget", type=int, choices=dataset_b.BUDGETS, required=True)
    evaluate.add_argument("--precision", choices=("nf4", "bf16_cpu_offload"), default="nf4")
    evaluate.add_argument("--baseline", type=Path, required=True)
    evaluate.add_argument("--deployment", type=Path, required=True)
    evaluate.add_argument("--batch", type=int, default=4)
    evaluate.set_defaults(run=evaluate_artifact)
    for command in (freeze, train, verdict):
        command.add_argument("--targets-index", default="index-B1-duplex.json")
    for command in (train, verdict):
        command.add_argument("--frame-offset", type=int, default=interface_fit.CACHE_FRAME_OFFSET)
    for command in (prep, cache, duplex_cache, target, audio, duplex, freeze, train, verdict,
                    artifact, evaluate):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda")
        command.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.run(args)


if __name__ == "__main__":
    main()
