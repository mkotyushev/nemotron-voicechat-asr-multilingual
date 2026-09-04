#!/usr/bin/env python3
"""Run Comparison 1: the unmodified PT_ML reference.

The runner consumes one already-frozen ``shared_setup.json``.  It attaches the
original FT_EN VoiceChat projection and featurizer to PT_ML without fitting a
map, exports and exactly reloads the F32 checkpoint, converts that final
artifact to the frozen Q8_0 deployment format, reloads the actual GGUF, runs the
common evaluator at both precision stages, and saves the PT_ML reference arrays
and residual-boundary activations needed by later comparisons.

The reserved LibriSpeech ``test`` split is verified as part of the frozen data
manifest but is never encoded or scored.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from asr_align import baseline, data, encoder as encoder_module, evaluation, export, features
from asr_align import hooks, manifests
from asr_align.experiments import ExperimentValidationError, sha256_file
from asr_align.weights import load_asr, load_mmproj, load_voicechat_safetensors

logger = logging.getLogger("pt-ml-baseline")
REPOSITORY = Path(__file__).resolve().parent
DEFAULT_WORK = REPOSITORY / ".cache" / "llama-voicechat.cpp"
EXPECTED_RUNTIME_READER_COMMIT = "f45001fc3d8013c72beb6753d3eb0b976b6a9fff"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    label = resolved.relative_to(relative_to.resolve()).as_posix() if relative_to else str(resolved)
    return {"path": label, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _tree_records(root: Path) -> list[dict[str, Any]]:
    return [_file_record(path, relative_to=root) for path in sorted(root.rglob("*")) if path.is_file()]


def _runtime_reader_provenance(work: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(work), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(work), "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExperimentValidationError(f"cannot inspect runtime reader at {work}: {error}") from error
    if commit != EXPECTED_RUNTIME_READER_COMMIT:
        raise ExperimentValidationError(
            f"runtime reader is at {commit}, expected {EXPECTED_RUNTIME_READER_COMMIT}"
        )
    sources = sorted((work / "gguf-py").rglob("*.py"))
    sources.append(work / "tools" / "voicechat" / "vc_gguf.py")
    patch = REPOSITORY / "patches" / "q8_0-converters.patch"
    return {
        "path": str(work),
        "commit": commit,
        "working_tree_changes": dirty,
        "source_files": [_file_record(path, relative_to=work) for path in sources],
        "q8_patch": _file_record(patch, relative_to=REPOSITORY),
    }


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _featurize_clips(
    clips: Sequence[data.Clip],
    mel_filters: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [features.log_mel(data.load(clip), mel_filters, window).float() for clip in clips]
    )


def _read_audio(path: Path) -> torch.Tensor:
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="float32")
    if rate != features.SAMPLE_RATE:
        raise ExperimentValidationError(
            f"{path}: {rate} Hz, but the frozen featurizer requires {features.SAMPLE_RATE} Hz"
        )
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return torch.from_numpy(samples)


def _write_activation_shard(
    path: Path,
    store: Mapping[str, torch.Tensor],
    *,
    manifest_sha256: str,
    split: str,
    records: Sequence[str],
) -> dict[str, Any]:
    if not store:
        raise ExperimentValidationError("activation hooks produced no residual-boundary tensors")
    arrays = {name: _as_numpy(value) for name, value in sorted(store.items())}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ExperimentValidationError("residual-boundary activation cache contains non-finite values")
    export.write_safetensors(
        path,
        arrays,
        {
            "comparison": "1",
            "candidate_id": baseline.BASELINE_CANDIDATE_ID,
            "precision": "pre_quantization",
            "manifest_sha256": manifest_sha256,
            "split": split,
            "records": json.dumps(list(records), separators=(",", ":")),
        },
    )
    return {
        **_file_record(path, relative_to=path.parents[2]),
        "split": split,
        "records": list(records),
        "tensors": {name: list(value.shape) for name, value in arrays.items()},
    }


@torch.inference_mode()
def _cache_map_train_activations(
    model: torch.nn.Module,
    clips: Sequence[data.Clip],
    manifest_records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    output: Path,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    store: dict[str, torch.Tensor] = {}
    handles = hooks.register(model, store, (hooks.RESIDUAL_GROUP,))
    rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(clips), batch_size):
            stop = min(start + batch_size, len(clips))
            store.clear()
            mel = _featurize_clips(clips[start:stop], mel_filters, window).to(device)
            hidden = model(mel)
            if not bool(torch.isfinite(hidden).all()):
                raise ExperimentValidationError("PT_ML map-train output contains NaN or infinity")
            paths = [str(record["path"]) for record in manifest_records[start:stop]]
            shard = output / "map_train" / f"batch-{start // batch_size:05d}.safetensors"
            rows.append(
                _write_activation_shard(
                    shard,
                    store,
                    manifest_sha256=manifest_sha256,
                    split="map_train",
                    records=paths,
                )
            )
            logger.info("map-train activations: %d/%d clips", stop, len(clips))
    finally:
        for handle in handles:
            handle.remove()
    return rows


@torch.inference_mode()
def _collect_librispeech_validation(
    pre_model: torch.nn.Module,
    post_model: torch.nn.Module,
    target_model: torch.nn.Module,
    clips: Sequence[data.Clip],
    manifest_records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    eval_frames: int,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    activation_output: Path,
    manifest_sha256: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    if eval_frames <= 0:
        raise ExperimentValidationError("--eval-frames must be positive")
    arrays: dict[str, list[np.ndarray]] = {
        "pre_hidden_pooled": [],
        "pre_voicechat_pooled": [],
        "post_hidden_pooled": [],
        "post_voicechat_pooled": [],
        "target_hidden_pooled": [],
        "target_voicechat_pooled": [],
        "pre_english_prediction": [],
        "post_english_prediction": [],
        "english_target": [],
    }
    kept_frames = 0
    store: dict[str, torch.Tensor] = {}
    handles = hooks.register(pre_model, store, (hooks.RESIDUAL_GROUP,))
    activation_rows: list[dict[str, Any]] = []
    try:
        for start in range(0, len(clips), batch_size):
            stop = min(start + batch_size, len(clips))
            mel = _featurize_clips(clips[start:stop], mel_filters, window).to(device)
            store.clear()
            pre_hidden = pre_model(mel)
            post_hidden = post_model(mel)
            target_hidden = target_model(mel)
            for name, value in (
                ("pre", pre_hidden), ("post", post_hidden), ("FT_EN", target_hidden)
            ):
                if not bool(torch.isfinite(value).all()):
                    raise ExperimentValidationError(
                        f"{name} LibriSpeech validation output contains NaN or infinity"
                    )
            pre_projected = pre_model.project(pre_hidden)
            post_projected = post_model.project(post_hidden)
            target_projected = target_model.project(target_hidden)
            arrays["pre_hidden_pooled"].append(_as_numpy(pre_hidden.mean(dim=1)))
            arrays["pre_voicechat_pooled"].append(_as_numpy(pre_projected.mean(dim=1)))
            arrays["post_hidden_pooled"].append(_as_numpy(post_hidden.mean(dim=1)))
            arrays["post_voicechat_pooled"].append(_as_numpy(post_projected.mean(dim=1)))
            arrays["target_hidden_pooled"].append(_as_numpy(target_hidden.mean(dim=1)))
            arrays["target_voicechat_pooled"].append(_as_numpy(target_projected.mean(dim=1)))

            remaining = max(0, eval_frames - kept_frames)
            if remaining:
                count = min(remaining, pre_projected.shape[0] * pre_projected.shape[1])
                arrays["pre_english_prediction"].append(
                    _as_numpy(pre_projected.reshape(-1, pre_projected.shape[-1])[:count])
                )
                arrays["post_english_prediction"].append(
                    _as_numpy(post_projected.reshape(-1, post_projected.shape[-1])[:count])
                )
                arrays["english_target"].append(
                    _as_numpy(target_projected.reshape(-1, target_projected.shape[-1])[:count])
                )
                kept_frames += count

            paths = [str(record["path"]) for record in manifest_records[start:stop]]
            shard = activation_output / "validation" / f"batch-{start // batch_size:05d}.safetensors"
            activation_rows.append(
                _write_activation_shard(
                    shard,
                    store,
                    manifest_sha256=manifest_sha256,
                    split="validation",
                    records=paths,
                )
            )
            logger.info("validation: %d/%d clips", stop, len(clips))
    finally:
        for handle in handles:
            handle.remove()
    if kept_frames == 0:
        raise ExperimentValidationError("LibriSpeech validation produced no evaluation frames")
    return {name: np.concatenate(values, axis=0) for name, values in arrays.items()}, activation_rows


@torch.inference_mode()
def _pool_paths(
    model: torch.nn.Module,
    paths: Iterable[Path],
    *,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
) -> dict[Path, tuple[np.ndarray, np.ndarray]]:
    pooled: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    unique = sorted({path.resolve() for path in paths})
    for index, path in enumerate(unique):
        mel = features.log_mel(_read_audio(path), mel_filters, window).float()[None].to(device)
        hidden = model(mel)
        projected = model.project(hidden)
        if not bool(torch.isfinite(hidden).all() and torch.isfinite(projected).all()):
            raise ExperimentValidationError(f"non-finite FLEURS embedding for {path}")
        pooled[path] = (_as_numpy(hidden[0].mean(dim=0)), _as_numpy(projected[0].mean(dim=0)))
        if (index + 1) % 25 == 0 or index + 1 == len(unique):
            logger.info("FLEURS pooled: %d/%d recordings", index + 1, len(unique))
    return pooled


def _stack(
    cache: Mapping[Path, tuple[np.ndarray, np.ndarray]],
    paths: Sequence[Path],
    which: int,
) -> np.ndarray:
    return np.stack([cache[path.resolve()][which] for path in paths])


def _collect_fleurs(
    pre_model: torch.nn.Module,
    post_model: torch.nn.Module,
    target_model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
) -> dict[str, dict[str, np.ndarray]]:
    root = Path(str(payload["root"])).resolve()
    path_groups: dict[str, dict[str, list[Path]]] = {}
    candidate_paths: list[Path] = []
    target_paths: list[Path] = []
    for language in payload["languages"]:
        rows = payload["pairs"][language]
        groups = {
            "english_reference": [
                manifests.resolve_take(root, row["english_reference"]) for row in rows
            ],
            "english_query": [
                manifests.resolve_take(root, row["english_query"]) for row in rows
            ],
            "foreign_query": [
                manifests.resolve_take(root, row["foreign_query"]) for row in rows
            ],
        }
        path_groups[language] = groups
        candidate_paths.extend(
            groups["english_reference"] + groups["english_query"] + groups["foreign_query"]
        )
        target_paths.extend(
            groups["english_reference"] + groups["english_query"] + groups["foreign_query"]
        )

    logger.info("pooling pre-quantization PT_ML FLEURS recordings")
    pre = _pool_paths(
        pre_model, candidate_paths, device=device, mel_filters=mel_filters, window=window
    )
    logger.info("pooling post-quantization PT_ML FLEURS recordings")
    post = _pool_paths(
        post_model, candidate_paths, device=device, mel_filters=mel_filters, window=window
    )
    logger.info("pooling FT_EN FLEURS controls and references")
    target = _pool_paths(
        target_model, target_paths, device=device, mel_filters=mel_filters, window=window
    )

    result: dict[str, dict[str, np.ndarray]] = {}
    for language, groups in path_groups.items():
        result[language] = {
            "pre_foreign_hidden": _stack(pre, groups["foreign_query"], 0),
            "pre_foreign_voicechat": _stack(pre, groups["foreign_query"], 1),
            "pre_english_reference_hidden": _stack(pre, groups["english_reference"], 0),
            "pre_english_query_voicechat": _stack(pre, groups["english_query"], 1),
            "post_foreign_hidden": _stack(post, groups["foreign_query"], 0),
            "post_foreign_voicechat": _stack(post, groups["foreign_query"], 1),
            "post_english_reference_hidden": _stack(post, groups["english_reference"], 0),
            "post_english_query_voicechat": _stack(post, groups["english_query"], 1),
            "target_english_reference_voicechat": _stack(
                target, groups["english_reference"], 1
            ),
            "target_english_query_voicechat": _stack(target, groups["english_query"], 1),
            "target_foreign_voicechat": _stack(target, groups["foreign_query"], 1),
        }
    return result


def _evaluate_stage(
    stage: str,
    libri: Mapping[str, np.ndarray],
    fleurs: Mapping[str, Mapping[str, np.ndarray]],
    *,
    manifest_hashes: Mapping[str, str],
    seed: int,
) -> dict[str, Any]:
    prefix = "pre" if stage == "pre_quantization" else "post"
    candidate_hidden = libri[f"{prefix}_hidden_pooled"]
    retrieval_inputs: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "candidate_on_english_retrieval": {},
        "historical_centered_fleurs_retrieval": {},
        "intrinsic_candidate_crosslingual_retrieval": {},
    }
    diagnostics: dict[str, np.ndarray] = {
        "librispeech_validation_candidate_voicechat_frames": libri[
            f"{prefix}_english_prediction"
        ],
        "librispeech_validation_candidate_hidden_pooled": candidate_hidden,
        "librispeech_validation_target_voicechat_frames": libri["english_target"],
    }
    for language, values in fleurs.items():
        english = values[f"{prefix}_english_query_voicechat"]
        historical = values[f"{prefix}_foreign_voicechat"]
        intrinsic = values[f"{prefix}_foreign_hidden"]
        retrieval_inputs["candidate_on_english_retrieval"][language] = (
            english,
            values["target_english_reference_voicechat"],
            english,
        )
        retrieval_inputs["historical_centered_fleurs_retrieval"][language] = (
            historical,
            values["target_english_reference_voicechat"],
            historical,
        )
        retrieval_inputs["intrinsic_candidate_crosslingual_retrieval"][language] = (
            intrinsic,
            values[f"{prefix}_english_reference_hidden"],
            intrinsic,
        )
        diagnostics[f"fleurs_{language}_candidate_foreign_hidden"] = intrinsic
        diagnostics[f"fleurs_{language}_candidate_foreign_voicechat"] = historical
        diagnostics[f"fleurs_{language}_candidate_english_voicechat"] = english
    result = evaluation.evaluate_candidate(
        comparison=1,
        candidate_id=baseline.BASELINE_CANDIDATE_ID,
        weight=baseline.BASELINE_LAMBDA,
        precision=stage,
        english_prediction=libri[f"{prefix}_english_prediction"],
        english_target=libri["english_target"],
        pt_ml_english_prediction=libri[f"{prefix}_english_prediction"],
        retrieval_inputs=retrieval_inputs,
        diagnostic_embeddings=diagnostics,
        manifest_hashes=manifest_hashes,
        seed=seed,
    )
    baseline.assert_zero_reference_deltas(result)
    return result


def _historical_reproduction(
    fleurs: Mapping[str, Mapping[str, np.ndarray]], *, seed: int, manifest_sha256: str
) -> dict[str, Any]:
    rows = []
    for language, values in fleurs.items():
        reference = values["target_english_reference_voicechat"]
        probes = {
            "en (second take)": values["target_english_query_voicechat"],
            "FT_EN/full_precision": values["target_foreign_voicechat"],
            "multilingual/pre_quantization": values["pre_foreign_voicechat"],
            "multilingual/post_quantization": values["post_foreign_voicechat"],
        }
        for name, probe in probes.items():
            rows.append(
                {
                    "language": language,
                    "probe": name,
                    **evaluation.retrieval_metrics(
                        probe, reference, pt_ml_probe=probe, centered=True, seed=seed
                    ),
                    "chance_top1": 1.0 / probe.shape[0],
                }
            )
    return {
        "schema_version": evaluation.RESULT_SCHEMA_VERSION,
        "evaluation": "historical_centered_fleurs_retrieval",
        "manifest_sha256": manifest_sha256,
        "rows": rows,
    }


def _embedding_bundle(
    libri: Mapping[str, np.ndarray], fleurs: Mapping[str, Mapping[str, np.ndarray]]
) -> dict[str, np.ndarray]:
    arrays = {f"librispeech.validation.{name}": value for name, value in libri.items()}
    for language, values in fleurs.items():
        arrays.update({f"fleurs.{language}.{name}": value for name, value in values.items()})
    return arrays


def _run_command(command: Sequence[str]) -> None:
    logger.info("running: %s", " ".join(command))
    subprocess.run(list(command), check=True, cwd=REPOSITORY)


def _convert_to_q8(checkpoint: Path, output: Path, work: Path) -> None:
    command = [
        sys.executable,
        str(REPOSITORY / "convert_asr_to_mmproj.py"),
        "--asr-dir",
        str(checkpoint),
        "--work",
        str(work),
        "--quant",
        "Q8_0",
        "--output",
        str(output),
    ]
    _run_command(command)


def _runtime_parity(
    checkpoint: Path,
    *,
    wav: Path | None,
    runtime_log: Path | None,
    device: torch.device,
    work: Path,
) -> dict[str, Any]:
    if (wav is None) != (runtime_log is None):
        raise ExperimentValidationError("--parity-wav and --runtime-log must be supplied together")
    if wav is None or runtime_log is None:
        return {"status": "not_requested"}
    command = [
        sys.executable,
        str(REPOSITORY / "check_encoder_parity.py"),
        "--asr-dir",
        str(checkpoint),
        "--wav",
        str(wav.resolve()),
        "--runtime-log",
        str(runtime_log.resolve()),
        "--work",
        str(work),
        "--device",
        str(device),
    ]
    _run_command(command)
    return {
        "status": "passed",
        "wav": _file_record(wav),
        "runtime_log": _file_record(runtime_log),
        "command": command,
    }


def run(args: argparse.Namespace) -> Path:
    shared = baseline.load_shared_setup(args.shared_setup, verify_checkpoint_hashes=True)
    work = args.work.resolve()
    if not (work / "gguf-py").is_dir() or not (work / "tools" / "voicechat").is_dir():
        raise ExperimentValidationError(
            f"{work} is not the prepared runtime reader; run align_setup.sh once or pass --work"
        )
    runtime_reader = _runtime_reader_provenance(work)
    output = args.output.resolve()
    if output.exists():
        raise ExperimentValidationError(
            f"refusing to replace baseline output {output}; choose a new experiment directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(shared.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(shared.seed)
    torch.use_deterministic_algorithms(True)

    logger.info("loading M/PT_ML without deployment rounding: %s", shared.pt_ml_path)
    pt_ml = load_asr(shared.pt_ml_path, mmproj_precision=False)
    logger.info(
        "loading full-precision F/FT_EN projection, featurizer, and target: %s",
        shared.ft_en_path,
    )
    ft_en = load_voicechat_safetensors(shared.ft_en_path)
    attached = baseline.attach_voicechat_interface(pt_ml, ft_en)
    encoder_keys = [key for key in pt_ml if key.startswith("encoder.")]
    original_encoder_equality = baseline.assert_exact_tensors(pt_ml, attached, keys=encoder_keys)
    original_interface_equality = baseline.assert_exact_tensors(
        ft_en, attached, keys=baseline.ATTACHED_KEYS
    )
    if attached.config != pt_ml.config:
        raise ExperimentValidationError("baseline runtime configuration differs from PT_ML")
    mel_filters = attached["featurizer.fb"]
    window = attached["featurizer.window"]
    del pt_ml

    libri_payload = manifests.load_manifest(shared.librispeech_manifest)
    fleurs_payload = manifests.load_manifest(shared.fleurs_manifest)
    manifests.verify_audio_files(fleurs_payload, root=Path(str(fleurs_payload["root"])))
    frozen_clips = data.from_frozen_manifest(shared.librispeech_manifest)
    validation_clips = frozen_clips["validation"]
    if not validation_clips:
        raise ExperimentValidationError("frozen LibriSpeech validation split is empty")

    stage_root = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        copied_setup = stage_root / "shared_setup.json"
        shutil.copyfile(shared.path, copied_setup)
        if sha256_file(copied_setup) != shared.sha256:
            raise ExperimentValidationError("copied shared_setup.json changed bytes")
        artifact = stage_root / "artifact" / baseline.BASELINE_CANDIDATE_ID
        command = [str(Path(sys.executable).resolve()), *sys.argv]
        report: dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_kind": "pt_ml_baseline",
            "comparison": 1,
            "candidate_id": baseline.BASELINE_CANDIDATE_ID,
            "map": None,
            "projection_dim": int(attached["proj.weight"].shape[0]),
            "source": str(shared.pt_ml_path),
            "ft_en_interface_source": str(shared.ft_en_path),
            "shared_setup": str(shared.path),
            "shared_setup_sha256": shared.sha256,
            "manifests": shared.manifest_hashes,
            "runtime_configuration_source": "M/PT_ML",
            "runtime_configuration_exact": True,
            "command": command,
        }
        export.export(
            artifact,
            source=shared.pt_ml_path,
            encoder={
                key: _as_numpy(value) for key, value in attached.items() if key.startswith("encoder.")
            },
            proj_weight=_as_numpy(attached["proj.weight"]),
            proj_bias=_as_numpy(attached["proj.bias"]),
            featurizer={
                "fb": _as_numpy(attached["featurizer.fb"]),
                "window": _as_numpy(attached["featurizer.window"]),
            },
            report=report,
        )
        reloaded = load_asr(artifact, mmproj_precision=False)
        pass_through_equality = baseline.assert_exact_tensors(attached, reloaded)
        if reloaded.config != attached.config:
            raise ExperimentValidationError("export changed the complete PT_ML encoder configuration")
        source_config = json.loads((shared.pt_ml_path / "config.json").read_text(encoding="utf-8"))
        artifact_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
        recorded_metadata = {
            "derived_from": artifact_config.pop("derived_from", None),
            "voicechat_baseline": artifact_config.pop("voicechat_baseline", None),
        }
        if artifact_config != source_config:
            raise ExperimentValidationError(
                "pass-through export changed PT_ML configuration fields other than provenance"
            )
        if recorded_metadata["voicechat_baseline"] is None:
            raise ExperimentValidationError("pass-through export omitted VoiceChat baseline provenance")

        first_mel = _featurize_clips(
            validation_clips[:1], mel_filters, window
        ).to(device)
        original_model = encoder_module.build(attached).to(device)
        sanity, original_hidden, original_projected = baseline.deterministic_forward_check(
            original_model, first_mel
        )
        pre_model = encoder_module.build(reloaded).to(device)
        reload_sanity, reload_hidden, reload_projected = baseline.deterministic_forward_check(
            pre_model, first_mel
        )
        export_hidden_change = baseline.numeric_change(original_hidden, reload_hidden)
        export_projected_change = baseline.numeric_change(original_projected, reload_projected)
        if export_hidden_change["changed_values"] or export_projected_change["changed_values"]:
            raise ExperimentValidationError("pass-through export changed the sanity-check output")
        del (
            attached,
            original_model,
            original_hidden,
            original_projected,
            reload_hidden,
            reload_projected,
        )

        report["checks"] = {
            "original_pt_ml_encoder": original_encoder_equality,
            "original_ft_en_interface": original_interface_equality,
            "export_reload": pass_through_equality,
            "source_configuration_preserved": True,
            "added_configuration_metadata": recorded_metadata,
            "sanity": sanity,
            "reload_sanity": reload_sanity,
            "export_hidden_change": export_hidden_change,
            "export_projected_change": export_projected_change,
        }
        _write_json(artifact / "baseline.json", report)

        deployment = stage_root / "deployment" / f"{baseline.BASELINE_CANDIDATE_ID}-Q8_0.gguf"
        _convert_to_q8(artifact, deployment, work)
        actual_post = load_mmproj(deployment, work, config=reloaded.config)
        simulated_post = load_asr(artifact, mmproj_precision=True)
        simulation_equality = baseline.assert_exact_tensors(simulated_post, actual_post)
        post_model = encoder_module.build(actual_post).to(device)
        post_sanity, post_hidden, post_projected = baseline.deterministic_forward_check(
            post_model, first_mel
        )
        quantized_hidden_change = baseline.numeric_change(
            pre_model(first_mel), post_hidden
        )
        quantized_projected_change = baseline.numeric_change(
            pre_model.project(pre_model(first_mel)), post_projected
        )
        weight_quantization = baseline.quantization_report(reloaded, actual_post)
        parity = _runtime_parity(
            artifact,
            wav=args.parity_wav,
            runtime_log=args.runtime_log,
            device=device,
            work=work,
        )
        del actual_post, reloaded, simulated_post, post_hidden, post_projected

        target_model = encoder_module.build(ft_en).to(device)
        del ft_en
        activation_root = stage_root / "activations" / baseline.BASELINE_CANDIDATE_ID
        libri_arrays, validation_activations = _collect_librispeech_validation(
            pre_model,
            post_model,
            target_model,
            validation_clips,
            libri_payload["splits"]["validation"],
            batch_size=args.batch,
            eval_frames=args.eval_frames,
            device=device,
            mel_filters=mel_filters,
            window=window,
            activation_output=activation_root,
            manifest_sha256=shared.manifest_hashes["librispeech"],
        )
        map_train_activations = _cache_map_train_activations(
            pre_model,
            frozen_clips["map_train"],
            libri_payload["splits"]["map_train"],
            batch_size=args.batch,
            device=device,
            mel_filters=mel_filters,
            window=window,
            output=activation_root,
            manifest_sha256=shared.manifest_hashes["librispeech"],
        )
        activation_index = {
            "schema_version": "1.0",
            "comparison": 1,
            "candidate_id": baseline.BASELINE_CANDIDATE_ID,
            "precision": "pre_quantization",
            "manifest_sha256": shared.manifest_hashes["librispeech"],
            "hook_points": "subsampling output and every block output",
            "reserved_test_encoded": False,
            "splits": {
                "map_train": map_train_activations,
                "validation": validation_activations,
            },
        }
        activation_index_path = activation_root / "index.json"
        _write_json(activation_index_path, activation_index)

        fleurs_arrays = _collect_fleurs(
            pre_model,
            post_model,
            target_model,
            fleurs_payload,
            device=device,
            mel_filters=mel_filters,
            window=window,
        )
        pre_result = _evaluate_stage(
            "pre_quantization",
            libri_arrays,
            fleurs_arrays,
            manifest_hashes=shared.manifest_hashes,
            seed=shared.seed,
        )
        post_result = _evaluate_stage(
            "post_quantization",
            libri_arrays,
            fleurs_arrays,
            manifest_hashes=shared.manifest_hashes,
            seed=shared.seed,
        )
        evaluation.validate_precision_pair(pre_result, post_result)
        precision_delta = baseline.precision_metric_delta(pre_result, post_result)
        historical = _historical_reproduction(
            fleurs_arrays,
            seed=shared.seed,
            manifest_sha256=shared.manifest_hashes["fleurs"],
        )

        result_paths = {
            "pre_quantization": stage_root / "results" / "pre_quantization.json",
            "post_quantization": stage_root / "results" / "post_quantization.json",
            "precision_delta": stage_root / "results" / "precision_delta.json",
            "historical_fleurs": stage_root / "results" / "historical_fleurs.json",
        }
        evaluation.write_result(result_paths["pre_quantization"], pre_result)
        evaluation.write_result(result_paths["post_quantization"], post_result)
        _write_json(result_paths["precision_delta"], precision_delta)
        _write_json(result_paths["historical_fleurs"], historical)

        embeddings_path = stage_root / "embeddings" / "pt_ml_reference.safetensors"
        export.write_safetensors(
            embeddings_path,
            _embedding_bundle(libri_arrays, fleurs_arrays),
            {
                "comparison": "1",
                "candidate_id": baseline.BASELINE_CANDIDATE_ID,
                "librispeech_manifest_sha256": shared.manifest_hashes["librispeech"],
                "fleurs_manifest_sha256": shared.manifest_hashes["fleurs"],
                "precision_stages": json.dumps(list(evaluation.PRECISION_STAGES)),
            },
        )

        run_report = {
            "schema_version": "1.0",
            "comparison": 1,
            "candidate_id": baseline.BASELINE_CANDIDATE_ID,
            "status": "complete",
            "command": command,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "working_directory": str(REPOSITORY),
                "deterministic_algorithms": True,
                "seed": shared.seed,
            },
            "shared_setup": {
                "source": str(shared.path),
                "copied_path": "shared_setup.json",
                "sha256": shared.sha256,
                "manifests": shared.manifest_hashes,
            },
            "sources": {
                role: {
                    "repo_id": record["repo_id"],
                    "revision": record["revision"],
                    "files": record["files"],
                }
                for role, record in shared.value["checkpoints"].items()
            },
            "runtime_reader": runtime_reader,
            "pass_through": report["checks"],
            "quantization": {
                "policy": shared.deployment_quantization,
                "artifact": _file_record(deployment, relative_to=stage_root),
                "converter_command": [
                    sys.executable,
                    "convert_asr_to_mmproj.py",
                    "--asr-dir",
                    "artifact/pt_ml-baseline",
                    "--work",
                    str(work),
                    "--quant",
                    "Q8_0",
                    "--output",
                    "deployment/pt_ml-baseline-Q8_0.gguf",
                ],
                "actual_artifact_matches_rounding_model": simulation_equality,
                "weights": weight_quantization,
                "sanity": post_sanity,
                "hidden_change": quantized_hidden_change,
                "projected_change": quantized_projected_change,
                "runtime_parity": parity,
            },
            "artifact": {
                "path": "artifact/pt_ml-baseline",
                "files": _tree_records(artifact),
            },
            "evaluation": {
                "split": "LibriSpeech/validation and frozen FLEURS",
                "english_frame_cap": args.eval_frames,
                "reserved_librispeech_test_encoded": False,
                "task_definitions": {
                    "english_voicechat_space": (
                        "PT_ML and FT_EN frames from LibriSpeech/validation, both through "
                        "the original FT_EN projection"
                    ),
                    "candidate_on_english_retrieval": (
                        "PT_ML English query take against FT_EN English reference take"
                    ),
                    "historical_centered_fleurs_retrieval": (
                        "PT_ML foreign take against FT_EN English reference take in VoiceChat space"
                    ),
                    "intrinsic_candidate_crosslingual_retrieval": (
                        "PT_ML foreign take against PT_ML English reference take before projection"
                    ),
                },
                "results": {
                    name: _file_record(path, relative_to=stage_root)
                    for name, path in result_paths.items()
                },
            },
            "reference_cache": {
                "embeddings": _file_record(embeddings_path, relative_to=stage_root),
                "activations": _file_record(activation_index_path, relative_to=stage_root),
                "activation_shards": len(map_train_activations) + len(validation_activations),
            },
        }
        run_path = stage_root / "run.json"
        _write_json(run_path, run_report)
        (stage_root / "run.json.sha256").write_text(
            sha256_file(run_path) + "  run.json\n", encoding="utf-8"
        )
        stage_root.rename(output)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    logger.info("Comparison 1 complete: %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-setup",
        type=Path,
        required=True,
        help="frozen shared_setup.json, or the directory containing it",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--eval-frames",
        type=int,
        default=24000,
        help="deterministic cap on validation frames retained for VoiceChat-space scoring",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parity-wav", type=Path, default=None)
    parser.add_argument("--runtime-log", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    try:
        run(args)
    except (ExperimentValidationError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Comparison 1 rejected: {error}") from error


if __name__ == "__main__":
    main()
