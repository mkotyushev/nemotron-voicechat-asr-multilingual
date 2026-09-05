#!/usr/bin/env python3
"""Run Comparison 2: direct encoder-only task arithmetic.

The runner consumes the frozen shared setup and the completed Comparison 1
cache.  It computes ``delta_F = FT_EN - PT_EN`` strictly in F32, constructs the
five fixed ``PT_ML + lambda * delta_F`` candidates, attaches the unchanged
FT_EN VoiceChat projection, evaluates every pre-quantization candidate against
the exact frozen PT_ML arrays, and converts/evaluates only the primary
``lambda=1`` deployment artifact at Q8_0.

The reserved LibriSpeech test split is verified by the manifest consumer but is
never encoded or scored.  No lambda is selected by this comparison.
"""

from __future__ import annotations

import argparse
import gc
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

import pt_ml_baseline as baseline_runner
from asr_align import baseline, data, direct, encoder as encoder_module
from asr_align import evaluation, export, features, hooks, manifests
from asr_align.experiments import (
    LAMBDAS,
    PRIMARY_LAMBDA,
    ExperimentValidationError,
    assert_runtime_config_inherited,
    candidate,
    inherit_runtime_config,
    sha256_file,
    task_vector,
    validate_encoder_triplet,
)
from asr_align.weights import (
    EncoderWeights,
    load_asr,
    load_mmproj,
    load_voicechat_safetensors,
)
from convert_asr_to_mmproj import SafeTensors

logger = logging.getLogger("direct-task-arithmetic")
REPOSITORY = Path(__file__).resolve().parent


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    label = resolved.relative_to(relative_to.resolve()).as_posix() if relative_to else str(resolved)
    return {"path": label, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _tree_records(root: Path) -> list[dict[str, Any]]:
    return [_file_record(path, relative_to=root) for path in sorted(root.rglob("*")) if path.is_file()]


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def _featurize_clips(
    clips: Sequence[data.Clip], mel_filters: torch.Tensor, window: torch.Tensor
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


def _read_reference_bundle(
    reference: direct.BaselineReference,
    stage: str,
    languages: Sequence[str],
) -> dict[str, Any]:
    """Read one precision stage from the content-verified Comparison 1 cache."""

    if stage not in evaluation.PRECISION_STAGES:
        raise ExperimentValidationError(f"unknown precision stage {stage}")
    prefix = "pre" if stage == "pre_quantization" else "post"
    source = SafeTensors(reference.embeddings_path)
    try:
        def take(name: str) -> np.ndarray:
            if name not in source:
                raise ExperimentValidationError(f"baseline embedding cache lacks {name}")
            value = np.array(source.f32(name), dtype=np.float32, order="C", copy=True)
            if value.ndim != 2 or not np.isfinite(value).all():
                raise ExperimentValidationError(f"baseline embedding {name} is not a finite matrix")
            return value

        libri = {
            "english_prediction": take(
                f"librispeech.validation.{prefix}_english_prediction"
            ),
            "hidden_pooled": take(f"librispeech.validation.{prefix}_hidden_pooled"),
            "voicechat_pooled": take(
                f"librispeech.validation.{prefix}_voicechat_pooled"
            ),
            "english_target": take("librispeech.validation.english_target"),
        }
        fleurs_bundle: dict[str, dict[str, np.ndarray]] = {}
        for language in languages:
            base = f"fleurs.{language}."
            fleurs_bundle[language] = {
                "english_query_voicechat": take(
                    base + f"{prefix}_english_query_voicechat"
                ),
                "foreign_voicechat": take(base + f"{prefix}_foreign_voicechat"),
                "foreign_hidden": take(base + f"{prefix}_foreign_hidden"),
                "english_reference_hidden": take(
                    base + f"{prefix}_english_reference_hidden"
                ),
                "target_english_reference_voicechat": take(
                    base + "target_english_reference_voicechat"
                ),
            }
    finally:
        source.f.close()
    return {"librispeech": libri, "fleurs": fleurs_bundle}


def _candidate_bundle_from_reference(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse exact frozen arrays for lambda=0 after tensor/config equality passes."""

    libri = reference["librispeech"]
    return {
        "librispeech": {
            "english_prediction": libri["english_prediction"],
            "hidden_pooled": libri["hidden_pooled"],
            "voicechat_pooled": libri["voicechat_pooled"],
        },
        "fleurs": {
            language: {
                key: values[key]
                for key in (
                    "english_query_voicechat",
                    "foreign_voicechat",
                    "foreign_hidden",
                    "english_reference_hidden",
                )
            }
            for language, values in reference["fleurs"].items()
        },
    }


@torch.inference_mode()
def _forward_check(
    model: torch.nn.Module, mel: torch.Tensor
) -> tuple[dict[str, Any], dict[str, Any]]:
    store: dict[str, torch.Tensor] = {}
    handles = hooks.register(model, store, (hooks.RESIDUAL_GROUP,))
    try:
        sanity, _, projected = baseline.deterministic_forward_check(model, mel)
        store["voicechat.projection"] = projected
        profile = direct.activation_profile(store)
    finally:
        for handle in handles:
            handle.remove()
    return sanity, profile


@torch.inference_mode()
def _collect_librispeech(
    model: torch.nn.Module,
    clips: Sequence[data.Clip],
    *,
    batch_size: int,
    eval_frames: int,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    candidate_name: str,
) -> dict[str, np.ndarray]:
    if eval_frames <= 0:
        raise ExperimentValidationError("baseline English frame cap must be positive")
    arrays: dict[str, list[np.ndarray]] = {
        "hidden_pooled": [],
        "voicechat_pooled": [],
        "english_prediction": [],
    }
    kept_frames = 0
    for start in range(0, len(clips), batch_size):
        stop = min(start + batch_size, len(clips))
        mel = _featurize_clips(clips[start:stop], mel_filters, window).to(device)
        hidden = model(mel)
        projected = model.project(hidden)
        if not bool(torch.isfinite(hidden).all() and torch.isfinite(projected).all()):
            raise ExperimentValidationError(
                f"{candidate_name} LibriSpeech validation output contains NaN or infinity"
            )
        arrays["hidden_pooled"].append(_as_numpy(hidden.mean(dim=1)))
        arrays["voicechat_pooled"].append(_as_numpy(projected.mean(dim=1)))
        remaining = max(0, eval_frames - kept_frames)
        if remaining:
            count = min(remaining, projected.shape[0] * projected.shape[1])
            arrays["english_prediction"].append(
                _as_numpy(projected.reshape(-1, projected.shape[-1])[:count])
            )
            kept_frames += count
        logger.info("%s LibriSpeech validation: %d/%d clips", candidate_name, stop, len(clips))
    if kept_frames == 0:
        raise ExperimentValidationError("LibriSpeech validation produced no evaluation frames")
    return {name: np.concatenate(values, axis=0) for name, values in arrays.items()}


@torch.inference_mode()
def _pool_paths(
    model: torch.nn.Module,
    paths: Iterable[Path],
    *,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    candidate_name: str,
) -> dict[Path, tuple[np.ndarray, np.ndarray]]:
    pooled: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    unique = sorted({path.resolve() for path in paths})
    for index, path in enumerate(unique):
        mel = features.log_mel(_read_audio(path), mel_filters, window).float()[None].to(device)
        hidden = model(mel)
        projected = model.project(hidden)
        if not bool(torch.isfinite(hidden).all() and torch.isfinite(projected).all()):
            raise ExperimentValidationError(f"non-finite {candidate_name} embedding for {path}")
        pooled[path] = (
            _as_numpy(hidden[0].mean(dim=0)),
            _as_numpy(projected[0].mean(dim=0)),
        )
        if (index + 1) % 25 == 0 or index + 1 == len(unique):
            logger.info("%s FLEURS pooled: %d/%d recordings", candidate_name, index + 1, len(unique))
    return pooled


def _stack(
    cache: Mapping[Path, tuple[np.ndarray, np.ndarray]], paths: Sequence[Path], which: int
) -> np.ndarray:
    return np.stack([cache[path.resolve()][which] for path in paths])


def _collect_fleurs(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    candidate_name: str,
) -> dict[str, dict[str, np.ndarray]]:
    root = Path(str(payload["root"])).resolve()
    path_groups: dict[str, dict[str, list[Path]]] = {}
    candidate_paths: list[Path] = []
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
    pooled = _pool_paths(
        model,
        candidate_paths,
        device=device,
        mel_filters=mel_filters,
        window=window,
        candidate_name=candidate_name,
    )
    return {
        language: {
            "english_query_voicechat": _stack(pooled, groups["english_query"], 1),
            "foreign_voicechat": _stack(pooled, groups["foreign_query"], 1),
            "foreign_hidden": _stack(pooled, groups["foreign_query"], 0),
            "english_reference_hidden": _stack(pooled, groups["english_reference"], 0),
        }
        for language, groups in path_groups.items()
    }


def _evaluate_stage(
    *,
    candidate_name: str,
    weight: float,
    stage: str,
    candidate_bundle: Mapping[str, Any],
    reference_bundle: Mapping[str, Any],
    manifest_hashes: Mapping[str, str],
    seed: int,
) -> dict[str, Any]:
    candidate_libri = candidate_bundle["librispeech"]
    reference_libri = reference_bundle["librispeech"]
    retrieval_inputs: dict[str, dict[str, tuple[np.ndarray, ...]]] = {
        task: {} for task in evaluation.RETRIEVAL_TASKS
    }
    diagnostics: dict[str, np.ndarray] = {
        "librispeech_validation_candidate_voicechat_frames": candidate_libri[
            "english_prediction"
        ],
        "librispeech_validation_candidate_hidden_pooled": candidate_libri["hidden_pooled"],
        "librispeech_validation_target_voicechat_frames": reference_libri["english_target"],
    }
    for language, values in candidate_bundle["fleurs"].items():
        frozen = reference_bundle["fleurs"][language]
        retrieval_inputs["candidate_on_english_retrieval"][language] = (
            values["english_query_voicechat"],
            frozen["target_english_reference_voicechat"],
            frozen["english_query_voicechat"],
            frozen["target_english_reference_voicechat"],
        )
        retrieval_inputs["historical_centered_fleurs_retrieval"][language] = (
            values["foreign_voicechat"],
            frozen["target_english_reference_voicechat"],
            frozen["foreign_voicechat"],
            frozen["target_english_reference_voicechat"],
        )
        retrieval_inputs["intrinsic_candidate_crosslingual_retrieval"][language] = (
            values["foreign_hidden"],
            values["english_reference_hidden"],
            frozen["foreign_hidden"],
            frozen["english_reference_hidden"],
        )
        diagnostics[f"fleurs_{language}_candidate_foreign_hidden"] = values[
            "foreign_hidden"
        ]
        diagnostics[f"fleurs_{language}_candidate_foreign_voicechat"] = values[
            "foreign_voicechat"
        ]
        diagnostics[f"fleurs_{language}_candidate_english_voicechat"] = values[
            "english_query_voicechat"
        ]
    return evaluation.evaluate_candidate(
        comparison=direct.COMPARISON,
        candidate_id=candidate_name,
        weight=weight,
        precision=stage,
        english_prediction=candidate_libri["english_prediction"],
        english_target=reference_libri["english_target"],
        pt_ml_english_prediction=reference_libri["english_prediction"],
        retrieval_inputs=retrieval_inputs,
        diagnostic_embeddings=diagnostics,
        manifest_hashes=manifest_hashes,
        seed=seed,
    )


def _write_candidate_embeddings(
    path: Path,
    bundle: Mapping[str, Any],
    *,
    candidate_name: str,
    weight: float,
    stage: str,
    manifest_hashes: Mapping[str, str],
) -> None:
    arrays = {
        f"librispeech.validation.{name}": value
        for name, value in bundle["librispeech"].items()
    }
    for language, values in bundle["fleurs"].items():
        arrays.update({f"fleurs.{language}.{name}": value for name, value in values.items()})
    export.write_safetensors(
        path,
        arrays,
        {
            "comparison": str(direct.COMPARISON),
            "candidate_id": candidate_name,
            "lambda": str(weight),
            "precision": stage,
            "librispeech_manifest_sha256": manifest_hashes["librispeech"],
            "fleurs_manifest_sha256": manifest_hashes["fleurs"],
        },
    )


def _assert_lambda_zero_result(
    result: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    for field in ("manifests", "evaluations", "embedding_diagnostics"):
        if result.get(field) != reference.get(field):
            raise ExperimentValidationError(
                f"lambda=0 evaluator output differs from frozen PT_ML in {field}"
            )
    return {
        "exact": True,
        "comparison_1_candidate_id": reference["candidate_id"],
        "fields": ["manifests", "evaluations", "embedding_diagnostics"],
    }


def _write_pareto_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Comparison 2 development Pareto table",
        "",
        "No final lambda is selected here. All values are pre-quantization.",
        "",
        "| lambda | English R2 | English cosine | intrinsic MRR | intrinsic top-1 | VoiceChat cross-lingual MRR | Pareto |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {lambda:g} | {english_voicechat_r2:+.6f} | "
            "{english_voicechat_cosine:.6f} | {multilingual_intrinsic_mrr_macro:.6f} | "
            "{multilingual_intrinsic_top1_macro:.6f} | "
            "{voicechat_crosslingual_mrr_macro:.6f} | {pareto} |".format(
                **row, pareto="yes" if row["pareto_efficient"] else "no"
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _release_model(*values: Any) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> Path:
    shared = baseline.load_shared_setup(args.shared_setup, verify_checkpoint_hashes=True)
    reference = direct.load_baseline_reference(args.baseline, shared)
    work = (
        args.work.resolve()
        if args.work is not None
        else Path(str(reference.run["runtime_reader"]["path"])).resolve()
    )
    if not (work / "gguf-py").is_dir() or not (work / "tools" / "voicechat").is_dir():
        raise ExperimentValidationError(f"{work} is not the prepared runtime reader")
    runtime_reader = baseline_runner._runtime_reader_provenance(work)

    output = args.output.resolve()
    if output.exists():
        raise ExperimentValidationError(
            f"refusing to replace Comparison 2 output {output}; choose a new experiment directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(shared.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(shared.seed)
    torch.use_deterministic_algorithms(True)

    checkpoints = shared.value["checkpoints"]
    pt_en_path = Path(str(checkpoints["E"]["path"])).resolve()
    logger.info("loading E/PT_EN without deployment rounding: %s", pt_en_path)
    pt_en = load_asr(pt_en_path, mmproj_precision=False)
    logger.info("loading M/PT_ML without deployment rounding: %s", shared.pt_ml_path)
    pt_ml = load_asr(shared.pt_ml_path, mmproj_precision=False)
    logger.info("loading original F/FT_EN safetensors: %s", shared.ft_en_path)
    ft_en = load_voicechat_safetensors(shared.ft_en_path)
    states = validate_encoder_triplet(pt_en, pt_ml, ft_en)
    delta = task_vector(states["E"], states["F"])
    task_report = direct.task_vector_report(states["E"], states["F"], delta)
    pt_ml_norm = direct.state_norm_report(states["M"])
    assert_runtime_config_inherited(inherit_runtime_config(pt_ml.config), pt_ml.config)

    libri_payload = manifests.load_manifest(shared.librispeech_manifest)
    fleurs_payload = manifests.load_manifest(shared.fleurs_manifest)
    manifests.verify_audio_files(fleurs_payload, root=Path(str(fleurs_payload["root"])))
    frozen_clips = data.from_frozen_manifest(shared.librispeech_manifest)
    validation_clips = frozen_clips["validation"]
    if not validation_clips:
        raise ExperimentValidationError("frozen LibriSpeech validation split is empty")
    eval_frames = int(reference.run["evaluation"]["english_frame_cap"])
    mel_filters = ft_en["featurizer.fb"]
    window = ft_en["featurizer.window"]
    first_mel = _featurize_clips(validation_clips[:1], mel_filters, window).to(device)
    pre_reference = _read_reference_bundle(
        reference, "pre_quantization", fleurs_payload["languages"]
    )

    stage_root = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        copied_setup = stage_root / "shared_setup.json"
        shutil.copyfile(shared.path, copied_setup)
        if sha256_file(copied_setup) != shared.sha256:
            raise ExperimentValidationError("copied shared_setup.json changed bytes")
        task_report_path = stage_root / "analysis" / "task_vector_norms.json"
        _write_json(task_report_path, task_report)
        command = [str(Path(sys.executable).resolve()), *sys.argv]

        candidate_records: dict[str, Any] = {}
        pre_results: dict[float, dict[str, Any]] = {}
        lambda_zero_profile: dict[str, Any] | None = None
        primary_artifact: Path | None = None

        for weight in LAMBDAS:
            name = direct.candidate_id(weight)
            logger.info("constructing %s", name)
            encoder_state = candidate(states["M"], delta, weight)
            weight_norm = direct.state_norm_report(encoder_state)
            weight_norm["l2_ratio_vs_pt_ml"] = float(weight_norm["l2"]) / max(
                float(pt_ml_norm["l2"]), 1e-30
            )
            if weight_norm["l2_ratio_vs_pt_ml"] > direct.MAX_ACTIVATION_GROWTH:
                raise ExperimentValidationError(
                    f"{name} encoder L2 growth exceeds the safety tripwire"
                )
            candidate_weights = EncoderWeights(
                encoder_state,
                inherit_runtime_config(pt_ml.config),
                name,
            )
            attached = baseline.attach_voicechat_interface(candidate_weights, ft_en)
            assert_runtime_config_inherited(attached.config, pt_ml.config)
            interface_equality = baseline.assert_exact_tensors(
                ft_en, attached, keys=baseline.ATTACHED_KEYS
            )

            artifact = stage_root / "artifacts" / name
            artifact_report: dict[str, Any] = {
                "schema_version": "1.0",
                "artifact_kind": direct.ARTIFACT_KIND,
                "comparison": direct.COMPARISON,
                "candidate_id": name,
                "lambda": weight,
                "formula": "C_lambda = M + lambda * (F - E)",
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
                encoder={key: _as_numpy(value) for key, value in encoder_state.items()},
                proj_weight=_as_numpy(attached["proj.weight"]),
                proj_bias=_as_numpy(attached["proj.bias"]),
                featurizer={
                    "fb": _as_numpy(attached["featurizer.fb"]),
                    "window": _as_numpy(attached["featurizer.window"]),
                },
                report=artifact_report,
            )
            reloaded = load_asr(artifact, mmproj_precision=False)
            export_equality = baseline.assert_exact_tensors(attached, reloaded)
            assert_runtime_config_inherited(reloaded.config, pt_ml.config)
            source_config = json.loads(
                (shared.pt_ml_path / "config.json").read_text(encoding="utf-8")
            )
            artifact_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
            metadata = {
                "derived_from": artifact_config.pop("derived_from", None),
                "voicechat_direct_task_arithmetic": artifact_config.pop(
                    "voicechat_direct_task_arithmetic", None
                ),
            }
            if artifact_config != source_config or metadata[
                "voicechat_direct_task_arithmetic"
            ] is None:
                raise ExperimentValidationError(
                    f"{name} export changed PT_ML config fields or omitted provenance"
                )

            model = encoder_module.build(reloaded).to(device)
            sanity, activation_profile = _forward_check(model, first_mel)
            if weight == 0.0:
                lambda_zero_profile = activation_profile
            if lambda_zero_profile is None:  # pragma: no cover - LAMBDAS starts at zero
                raise ExperimentValidationError("lambda=0 must be evaluated first")
            growth = direct.activation_growth_report(lambda_zero_profile, activation_profile)

            lambda_zero_checks: dict[str, Any] | None = None
            if weight == 0.0:
                baseline_artifact = load_asr(reference.root / reference.run["artifact"]["path"], mmproj_precision=False)
                lambda_zero_checks = {
                    "tensors": baseline.assert_exact_tensors(baseline_artifact, reloaded),
                    "runtime_configuration_exact": baseline_artifact.config == reloaded.config,
                }
                if not lambda_zero_checks["runtime_configuration_exact"]:
                    raise ExperimentValidationError(
                        "lambda=0 runtime configuration differs from Comparison 1"
                    )
                candidate_bundle = _candidate_bundle_from_reference(pre_reference)
                del baseline_artifact
            else:
                candidate_bundle = {
                    "librispeech": _collect_librispeech(
                        model,
                        validation_clips,
                        batch_size=args.batch,
                        eval_frames=eval_frames,
                        device=device,
                        mel_filters=mel_filters,
                        window=window,
                        candidate_name=name,
                    ),
                    "fleurs": _collect_fleurs(
                        model,
                        fleurs_payload,
                        device=device,
                        mel_filters=mel_filters,
                        window=window,
                        candidate_name=name,
                    ),
                }
            result = _evaluate_stage(
                candidate_name=name,
                weight=weight,
                stage="pre_quantization",
                candidate_bundle=candidate_bundle,
                reference_bundle=pre_reference,
                manifest_hashes=shared.manifest_hashes,
                seed=shared.seed,
            )
            if weight == 0.0:
                lambda_zero_checks["evaluation"] = _assert_lambda_zero_result(
                    result, reference.results["pre_quantization"]
                )
            result_path = stage_root / "results" / name / "pre_quantization.json"
            evaluation.write_result(result_path, result)
            embeddings_path = stage_root / "embeddings" / name / "pre_quantization.safetensors"
            _write_candidate_embeddings(
                embeddings_path,
                candidate_bundle,
                candidate_name=name,
                weight=weight,
                stage="pre_quantization",
                manifest_hashes=shared.manifest_hashes,
            )
            artifact_report["checks"] = {
                "original_ft_en_interface": interface_equality,
                "export_reload": export_equality,
                "source_configuration_preserved": True,
                "added_configuration_metadata": metadata,
                "sanity": sanity,
                "activation_growth": growth,
                "lambda_zero": lambda_zero_checks,
            }
            artifact_report["weight_norm"] = weight_norm
            _write_json(artifact / "direct_task_arithmetic.json", artifact_report)
            candidate_records[name] = {
                "lambda": weight,
                "artifact": {"path": artifact.relative_to(stage_root).as_posix()},
                "weight_norm": weight_norm,
                "forward_check": {"sanity": sanity, "activation_growth": growth},
                "lambda_zero": lambda_zero_checks,
                "result": _file_record(result_path, relative_to=stage_root),
                "embeddings": _file_record(embeddings_path, relative_to=stage_root),
            }
            pre_results[weight] = result
            if weight == PRIMARY_LAMBDA:
                primary_artifact = artifact
            del candidate_bundle, model, reloaded, attached, candidate_weights, encoder_state
            _release_model()

        if primary_artifact is None or lambda_zero_profile is None:  # pragma: no cover
            raise ExperimentValidationError("the frozen sweep omitted a required endpoint")
        pareto = direct.pareto_table(list(pre_results.values()))
        pareto_json = stage_root / "analysis" / "pareto.json"
        pareto_markdown = stage_root / "analysis" / "pareto.md"
        _write_json(pareto_json, pareto)
        _write_pareto_markdown(pareto_markdown, pareto)

        primary_name = direct.candidate_id(PRIMARY_LAMBDA)
        deployment = stage_root / "deployment" / f"{primary_name}-Q8_0.gguf"
        baseline_runner._convert_to_q8(primary_artifact, deployment, work)
        primary_pre = load_asr(primary_artifact, mmproj_precision=False)
        actual_post = load_mmproj(deployment, work, config=primary_pre.config)
        simulated_post = load_asr(primary_artifact, mmproj_precision=True)
        simulation_equality = baseline.assert_exact_tensors(simulated_post, actual_post)
        quantization = baseline.quantization_report(primary_pre, actual_post)
        post_model = encoder_module.build(actual_post).to(device)
        post_sanity, post_profile = _forward_check(post_model, first_mel)
        post_growth = direct.activation_growth_report(lambda_zero_profile, post_profile)
        post_bundle = {
            "librispeech": _collect_librispeech(
                post_model,
                validation_clips,
                batch_size=args.batch,
                eval_frames=eval_frames,
                device=device,
                mel_filters=mel_filters,
                window=window,
                candidate_name=primary_name + "-Q8_0",
            ),
            "fleurs": _collect_fleurs(
                post_model,
                fleurs_payload,
                device=device,
                mel_filters=mel_filters,
                window=window,
                candidate_name=primary_name + "-Q8_0",
            ),
        }
        post_reference = _read_reference_bundle(
            reference, "post_quantization", fleurs_payload["languages"]
        )
        post_result = _evaluate_stage(
            candidate_name=primary_name,
            weight=PRIMARY_LAMBDA,
            stage="post_quantization",
            candidate_bundle=post_bundle,
            reference_bundle=post_reference,
            manifest_hashes=shared.manifest_hashes,
            seed=shared.seed,
        )
        evaluation.validate_precision_pair(pre_results[PRIMARY_LAMBDA], post_result)
        post_result_path = stage_root / "results" / primary_name / "post_quantization.json"
        evaluation.write_result(post_result_path, post_result)
        post_embeddings = stage_root / "embeddings" / primary_name / "post_quantization.safetensors"
        _write_candidate_embeddings(
            post_embeddings,
            post_bundle,
            candidate_name=primary_name,
            weight=PRIMARY_LAMBDA,
            stage="post_quantization",
            manifest_hashes=shared.manifest_hashes,
        )
        precision_delta = baseline.precision_metric_delta(
            pre_results[PRIMARY_LAMBDA], post_result
        )
        precision_delta_path = stage_root / "results" / primary_name / "precision_delta.json"
        _write_json(precision_delta_path, precision_delta)
        parity = baseline_runner._runtime_parity(
            primary_artifact,
            wav=args.parity_wav,
            runtime_log=args.runtime_log,
            device=device,
            work=work,
        )

        candidate_records[primary_name]["post_quantization"] = {
            "deployment": _file_record(deployment, relative_to=stage_root),
            "actual_artifact_matches_rounding_model": simulation_equality,
            "weight_change": quantization,
            "forward_check": {"sanity": post_sanity, "activation_growth": post_growth},
            "result": _file_record(post_result_path, relative_to=stage_root),
            "embeddings": _file_record(post_embeddings, relative_to=stage_root),
            "precision_delta": _file_record(precision_delta_path, relative_to=stage_root),
            "runtime_parity": parity,
        }
        del post_bundle, post_reference, post_model, actual_post, simulated_post, primary_pre
        _release_model()

        for name, record in candidate_records.items():
            artifact_path = stage_root / record["artifact"]["path"]
            record["artifact"]["files"] = _tree_records(artifact_path)

        run_report = {
            "schema_version": "1.0",
            "comparison": direct.COMPARISON,
            "artifact_kind": direct.ARTIFACT_KIND,
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
                for role, record in checkpoints.items()
            },
            "paired_pt_ml_reference": {
                "run": _file_record(reference.run_path),
                "embeddings": _file_record(reference.embeddings_path),
                "results": {
                    stage: _file_record(path) for stage, path in reference.result_paths.items()
                },
                "exact_frozen_arrays_used": True,
            },
            "runtime_reader": runtime_reader,
            "arithmetic": {
                "formula": "C_lambda = M + lambda * (F - E)",
                "precision": "float32",
                "lambdas": list(LAMBDAS),
                "primary_lambda": PRIMARY_LAMBDA,
                "task_vector_norms": _file_record(task_report_path, relative_to=stage_root),
                "lambda_selected": None,
            },
            "evaluation": {
                "split": "LibriSpeech/validation and frozen FLEURS",
                "english_frame_cap": eval_frames,
                "reserved_librispeech_test_encoded": False,
                "every_lambda_pre_quantization": True,
                "primary_lambda_post_quantization": True,
                "pareto": {
                    "json": _file_record(pareto_json, relative_to=stage_root),
                    "table": _file_record(pareto_markdown, relative_to=stage_root),
                    "selection_performed": False,
                },
            },
            "candidates": candidate_records,
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
    logger.info("Comparison 2 complete: %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-setup",
        type=Path,
        required=True,
        help="frozen shared_setup.json, or its containing directory",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="completed Comparison 1 output directory, or its run.json",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="prepared runtime reader (defaults to the exact path recorded by Comparison 1)",
    )
    parser.add_argument("--batch", type=int, default=4)
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
        raise SystemExit(f"Comparison 2 rejected: {error}") from error


if __name__ == "__main__":
    main()
