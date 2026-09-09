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
    english_gate_verdict,
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


def freeze_targets(args):
    experiment, data = load_experiment(args.output)
    entries, summary = [], {}
    for split, rows in data["splits"].items():
        for row in rows:
            pair = []
            for condition in gating.SYSTEM_PROMPTS:
                path = args.output / "targets" / row["pool"] / f"{row['clip_id'].replace('/', '-')}-{condition}.json"
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
    write_frozen(args.output / "targets" / "index.json", {
        "teacher_paths": data["teacher_paths"], "dataset_b_sha256": data["manifest_sha256"],
        "retention_rule": "retain a clip only if both prompt conditions pass the frozen gate",
        "teacher_quality": summary, "entries": entries,
        "provenance": {pool: record(args.output / "targets" / pool / "provenance.json") for pool in ("B1", "B2")}})


def load_training_items(root, data, targets, budget, arm, *, split="train"):
    from safetensors.torch import load_file
    allowed = set(dataset_b.budget_clip_ids(data, budget)) if split == "train" else {r["clip_id"] for r in data["splits"][split]}
    rows = {r["clip_id"]: r for r in data["splits"][split]}
    cache = read_frozen(root / "activations" / arm / "index.json")
    if cache["dataset_b_sha256"] != data["manifest_sha256"]:
        raise ExperimentValidationError("encoder cache uses another training manifest")
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
        timeline = target["timeline"] if rows[key]["pool"] == "B1" else text_target_timeline(target["reply_token_ids"], len(hidden))
        items.append({"clip_id": key, "condition": entry["condition"], "pool": rows[key]["pool"],
                      "language": rows[key]["language"], "features": hidden, "timeline": timeline})
    return items, cache


def fit(args):
    from safetensors.torch import load_file, save_file
    experiment, data = load_experiment(args.output)
    fitting_graph = validate_fitting_graph(experiment)
    targets = read_frozen(args.output / "targets/index.json")
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
    items, cache = load_training_items(args.output, data, targets, args.budget, args.arm)
    validation, _ = load_training_items(args.output, data, targets, args.budget, args.arm, split="validation")
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


def english_gate(args):
    """The control every other arm waits on: did fitting break the loop?"""

    from safetensors.torch import load_file

    experiment, data = load_experiment(args.output)
    validate_fitting_graph(experiment)
    targets = read_frozen(args.output / "targets/index.json")
    if targets["dataset_b_sha256"] != data["manifest_sha256"]:
        raise ExperimentValidationError("teacher targets use another Dataset B")
    fit_root, result, _ = read_fit(args.output, "E1", args.budget, args.precision)
    validation, cache = load_training_items(args.output, data, targets, args.budget, "E1", split="validation")
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
    # The initialization is E1's own untouched FT_EN projection, so this is the
    # design record's "trained proj against original proj on English" exactly.
    measured = {
        "initialization": scored_cells(linear_from(source["proj.weight"], source["proj.bias"], args.device),
                                       lm, english, prefixes, args.device),
        "fitted": scored_cells(linear_from(fitted["proj.weight"], fitted["proj.bias"], args.device),
                               lm, english, prefixes, args.device),
    }
    verdict = english_gate_verdict(measured["fitted"], measured["initialization"], tolerance=args.tolerance)
    value = write_frozen(args.output / "E1_english_gate.json",
                         {"arm": "E1", "budget": args.budget, "precision": args.precision,
                          "experiment_sha256": experiment["manifest_sha256"],
                          "dataset_b_sha256": data["manifest_sha256"],
                          "fit_result": record(fit_root / "result.json"),
                          "projection": dict(result["projection"]), "encoder": cache["encoder"],
                          "measured": measured, **verdict})
    LOG.info("E1 English gate %s: %+.5f nats against the initialization over %d held-out clip/prompt examples",
             "passed" if value["passed"] else "FAILED", value["delta"], value["clips_scored"])
    if not value["passed"]:
        raise ExperimentValidationError(
            "E1 did not match its initialization on English; the fitting loop is broken")


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
    freeze = sub.add_parser("freeze-targets")
    freeze.set_defaults(run=freeze_targets)
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
    for command in (prep, cache, target, audio, freeze, train, verdict, artifact, evaluate):
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
