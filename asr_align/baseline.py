"""Comparison 1 helpers for the unmodified ``PT_ML`` reference.

The baseline has deliberately little modelling logic: copy the canonical
``PT_ML`` encoder, attach the original ``FT_EN`` VoiceChat projection and
featurizer, and never fit an interface map.  This module keeps the strict,
cheap-to-test invariants separate from the CLI that performs the expensive
audio passes in :mod:`pt_ml_baseline`.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import evaluation, manifests
from .experiments import (
    EXPECTED_REPO_IDS,
    ROLE_DEFINITIONS,
    SHARED_SETUP_SCHEMA_VERSION,
    ExperimentValidationError,
    sha256_file,
)
from .weights import EncoderWeights

BASELINE_CANDIDATE_ID = "pt_ml-baseline"
BASELINE_LAMBDA = 0.0
REQUIRED_FLEURS_LANGUAGES = frozenset(("fr_fr", "de_de", "ru_ru"))
ATTACHED_KEYS = (
    "proj.weight",
    "proj.bias",
    "featurizer.fb",
    "featurizer.window",
)


@dataclass(frozen=True)
class SharedSetup:
    """Validated paths and hashes consumed by the Comparison 1 runner."""

    path: Path
    value: dict[str, Any]
    sha256: str
    pt_ml_path: Path
    ft_en_path: Path
    librispeech_manifest: Path
    fleurs_manifest: Path
    manifest_hashes: dict[str, str]
    seed: int
    deployment_quantization: str


def _resolved_setup_path(path: Path) -> Path:
    path = path.resolve()
    return path / "shared_setup.json" if path.is_dir() else path


def _verify_checkpoint_record(record: Mapping[str, Any]) -> None:
    for artifact in record["files"]:
        if not isinstance(artifact, Mapping):
            raise ExperimentValidationError("checkpoint file record is not an object")
        path = Path(str(artifact.get("path", "")))
        if not path.is_file():
            raise ExperimentValidationError(f"recorded checkpoint file is missing: {path}")
        try:
            expected_bytes = int(artifact.get("bytes", -1))
        except (TypeError, ValueError) as error:
            raise ExperimentValidationError(
                f"checkpoint file record has an invalid byte count: {path}"
            ) from error
        if path.stat().st_size != expected_bytes:
            raise ExperimentValidationError(
                f"checkpoint size changed for {path}: "
                f"recorded {expected_bytes}, found {path.stat().st_size}"
            )
        observed = sha256_file(path)
        if observed != artifact.get("sha256"):
            raise ExperimentValidationError(
                f"checkpoint SHA-256 changed for {path}: "
                f"recorded {artifact.get('sha256')}, found {observed}"
            )


def load_shared_setup(path: Path, *, verify_checkpoint_hashes: bool = True) -> SharedSetup:
    """Load the frozen setup and revalidate every Comparison 1 dependency.

    Checkpoint hashing can be disabled only for fast unit tests.  The CLI never
    disables it: a path alone is not sufficient provenance for an experiment.
    """

    setup_path = _resolved_setup_path(path)
    if not setup_path.is_file():
        raise ExperimentValidationError(f"shared setup does not exist: {setup_path}")
    value = json.loads(setup_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SHARED_SETUP_SCHEMA_VERSION:
        raise ExperimentValidationError(
            f"shared setup must have schema_version {SHARED_SETUP_SCHEMA_VERSION!r}"
        )
    if value.get("roles") != ROLE_DEFINITIONS:
        raise ExperimentValidationError("shared setup checkpoint roles changed")

    checkpoints = value.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(ROLE_DEFINITIONS):
        raise ExperimentValidationError("shared setup must record exactly E, M, and F")
    for role, record in checkpoints.items():
        if not isinstance(record, dict):
            raise ExperimentValidationError(f"shared setup {role} checkpoint record is invalid")
        if record.get("repo_id") != EXPECTED_REPO_IDS[role]:
            raise ExperimentValidationError(f"shared setup {role} repo identity changed")
        if record.get("name") != ROLE_DEFINITIONS[role]:
            raise ExperimentValidationError(f"shared setup {role} role name changed")
        expected_kind = "voicechat_safetensors" if role == "F" else "asr"
        if record.get("kind") != expected_kind:
            raise ExperimentValidationError(
                f"shared setup {role} must use {expected_kind}; quantized FT_EN sources are forbidden"
            )
        if not str(record.get("revision", "")).strip():
            raise ExperimentValidationError(f"shared setup {role} has no pinned revision")
        if not isinstance(record.get("files"), list) or not record["files"]:
            raise ExperimentValidationError(f"shared setup {role} has no hashed checkpoint files")
        if verify_checkpoint_hashes:
            _verify_checkpoint_record(record)

    runtime = value.get("candidate_runtime_configuration", {})
    if runtime.get("source") != "M/PT_ML" or runtime.get("exact_match") is not True:
        raise ExperimentValidationError("shared setup does not require exact PT_ML runtime config")
    if runtime.get("source_configuration_sha256") != checkpoints["M"].get(
        "configuration_sha256"
    ):
        raise ExperimentValidationError("shared setup PT_ML runtime configuration hash changed")

    precision = value.get("precision", {})
    quantization = str(precision.get("deployment_quantization", ""))
    if quantization != "Q8_0":
        raise ExperimentValidationError(
            "Comparison 1 currently requires the frozen Q8_0 deployment policy; "
            f"shared setup records {quantization!r}"
        )
    if precision.get("quantization_stage") != "final_artifact_only":
        raise ExperimentValidationError("shared setup no longer quantizes final artifacts only")
    if "no deployment quantization" not in str(precision.get("ft_en_source_note", "")):
        raise ExperimentValidationError("shared setup does not record an unquantized FT_EN source")

    recorded_manifests = value.get("manifests", {})
    if set(recorded_manifests) != {"librispeech", "fleurs"}:
        raise ExperimentValidationError("shared setup must record LibriSpeech and FLEURS manifests")
    libri_path = Path(str(recorded_manifests["librispeech"]["path"])).resolve()
    fleurs_path = Path(str(recorded_manifests["fleurs"]["path"])).resolve()
    libri = manifests.load_manifest(libri_path)
    fleurs = manifests.load_manifest(fleurs_path)
    manifests.validate_librispeech_manifest(libri)
    manifests.validate_fleurs_manifest(fleurs)
    hashes = {
        "librispeech": str(libri["manifest_sha256"]),
        "fleurs": str(fleurs["manifest_sha256"]),
    }
    for name, observed in hashes.items():
        recorded = recorded_manifests[name].get("sha256")
        if observed != recorded:
            raise ExperimentValidationError(
                f"{name} manifest differs from shared setup: recorded {recorded}, found {observed}"
            )
    missing_languages = REQUIRED_FLEURS_LANGUAGES - set(fleurs["languages"])
    if missing_languages:
        raise ExperimentValidationError(
            "Comparison 1 must reproduce the frozen French, German, and Russian probes; "
            f"missing {sorted(missing_languages)}"
        )
    if int(libri.get("seed", 0)) != int(fleurs.get("seed", 0)):
        raise ExperimentValidationError("the frozen LibriSpeech and FLEURS seeds differ")

    return SharedSetup(
        path=setup_path,
        value=value,
        sha256=sha256_file(setup_path),
        pt_ml_path=Path(str(checkpoints["M"]["path"])).resolve(),
        ft_en_path=Path(str(checkpoints["F"]["path"])).resolve(),
        librispeech_manifest=libri_path,
        fleurs_manifest=fleurs_path,
        manifest_hashes=hashes,
        seed=int(libri.get("seed", 0)),
        deployment_quantization=quantization,
    )


def attach_voicechat_interface(pt_ml: EncoderWeights, ft_en: EncoderWeights) -> EncoderWeights:
    """Return an independent PT_ML state with the untouched FT_EN interface.

    Only ``encoder.*`` tensors are admitted from PT_ML.  Projection and
    featurizer tensors can only come from FT_EN, and the complete PT_ML runtime
    configuration remains the source of graph parameters (including its
    56-frame left context).
    """

    unexpected = sorted(key for key in pt_ml if not key.startswith("encoder."))
    if unexpected:
        raise ExperimentValidationError(
            f"PT_ML contains unexpected non-encoder tensors: {unexpected[:8]}"
        )
    missing = [key for key in ATTACHED_KEYS if key not in ft_en]
    if missing:
        raise ExperimentValidationError(f"FT_EN interface tensors are missing: {missing}")
    tensors = {
        key: value.detach().clone().contiguous()
        for key, value in pt_ml.items()
        if key.startswith("encoder.")
    }
    if not tensors:
        raise ExperimentValidationError("PT_ML contains no canonical encoder tensors")
    for key in ATTACHED_KEYS:
        value = ft_en[key]
        if not bool(torch.isfinite(value).all()):
            raise ExperimentValidationError(f"FT_EN interface tensor {key} is not finite")
        tensors[key] = value.detach().clone().contiguous()
    return EncoderWeights(tensors, copy.deepcopy(pt_ml.config), pt_ml.name + "-baseline")


def assert_exact_tensors(
    expected: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
    *,
    keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Require exact keys, shapes, dtypes and values; never broadcast."""

    selected = sorted(keys if keys is not None else expected)
    expected_keys = set(selected)
    observed_keys = set(observed) if keys is None else {key for key in selected if key in observed}
    if expected_keys != observed_keys:
        raise ExperimentValidationError(
            "pass-through tensor keys differ: "
            f"missing={sorted(expected_keys - observed_keys)[:8]}, "
            f"extra={sorted(observed_keys - expected_keys)[:8]}"
        )
    total_values = 0
    for key in selected:
        left, right = expected[key], observed[key]
        if left.shape != right.shape:
            raise ExperimentValidationError(
                f"pass-through shape mismatch for {key}: {tuple(left.shape)} vs {tuple(right.shape)}"
            )
        if left.dtype != right.dtype:
            raise ExperimentValidationError(
                f"pass-through dtype mismatch for {key}: {left.dtype} vs {right.dtype}"
            )
        if not torch.equal(left, right):
            delta = float((left.double() - right.double()).abs().max())
            raise ExperimentValidationError(
                f"pass-through changed {key}; maximum absolute difference is {delta:g}"
            )
        total_values += left.numel()
    return {"exact": True, "tensor_count": len(selected), "value_count": total_values}


@torch.inference_mode()
def deterministic_forward_check(
    model: torch.nn.Module,
    mel: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Run the same input twice and require finite bit-identical outputs."""

    first_hidden = model(mel)
    second_hidden = model(mel)
    if not torch.equal(first_hidden, second_hidden):
        raise ExperimentValidationError("PT_ML forward pass is not deterministic")
    if not bool(torch.isfinite(first_hidden).all()):
        raise ExperimentValidationError("PT_ML forward pass produced NaN or infinity")
    if not hasattr(model, "project"):
        raise ExperimentValidationError("baseline model has no VoiceChat projection")
    first_projected = model.project(first_hidden)
    second_projected = model.project(second_hidden)
    if not torch.equal(first_projected, second_projected):
        raise ExperimentValidationError("VoiceChat projection is not deterministic")
    if not bool(torch.isfinite(first_projected).all()):
        raise ExperimentValidationError("VoiceChat projection produced NaN or infinity")
    report = {
        "deterministic": True,
        "finite": True,
        "input_shape": list(mel.shape),
        "hidden_shape": list(first_hidden.shape),
        "projected_shape": list(first_projected.shape),
        "hidden_mean": float(first_hidden.double().mean()),
        "hidden_rms": float(first_hidden.double().square().mean().sqrt()),
        "projected_mean": float(first_projected.double().mean()),
        "projected_rms": float(first_projected.double().square().mean().sqrt()),
    }
    return report, first_hidden, first_projected


def numeric_change(before: np.ndarray | torch.Tensor, after: np.ndarray | torch.Tensor) -> dict[str, Any]:
    """Summarize a same-shaped pre/post change without implicit broadcasting."""

    left = torch.as_tensor(before).detach().double().cpu()
    right = torch.as_tensor(after).detach().double().cpu()
    if left.shape != right.shape:
        raise ExperimentValidationError(
            f"pre/post shapes differ: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise ExperimentValidationError("pre/post comparison contains NaN or infinity")
    delta = right - left
    left_norm = float(torch.linalg.vector_norm(left))
    flat_left, flat_right = left.reshape(-1), right.reshape(-1)
    denominator = max(
        float(torch.linalg.vector_norm(flat_left) * torch.linalg.vector_norm(flat_right)),
        1e-30,
    )
    return {
        "shape": list(left.shape),
        "changed_values": int(torch.count_nonzero(delta)),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
        "rmse": float(delta.square().mean().sqrt()) if delta.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(delta)) / max(left_norm, 1e-30),
        "cosine": float(torch.dot(flat_left, flat_right)) / denominator,
    }


def quantization_report(
    pre: Mapping[str, torch.Tensor], post: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Measure Q8_0/F16 deployment rounding for every self-contained tensor."""

    if set(pre) != set(post):
        raise ExperimentValidationError("pre/post quantization tensor keys differ")
    rows: dict[str, Any] = {}
    changed_tensors = 0
    delta_sq = 0.0
    source_sq = 0.0
    max_abs = 0.0
    for key in sorted(pre):
        change = numeric_change(pre[key], post[key])
        rows[key] = change
        changed_tensors += int(change["changed_values"] > 0)
        difference = post[key].double() - pre[key].double()
        delta_sq += float(difference.square().sum())
        source_sq += float(pre[key].double().square().sum())
        max_abs = max(max_abs, float(change["max_abs"]))
    return {
        "tensor_count": len(rows),
        "changed_tensors": changed_tensors,
        "max_abs": max_abs,
        "relative_l2": math.sqrt(delta_sq) / max(math.sqrt(source_sq), 1e-30),
        "tensors": rows,
    }


def assert_zero_reference_deltas(result: Mapping[str, Any]) -> None:
    """Comparison 1 is its own paired reference at each precision stage."""

    evaluation.validate_result(result)
    groups: list[Mapping[str, Any]] = [
        result["evaluations"]["english_voicechat_space"]
    ]
    groups.extend(
        metrics
        for task in evaluation.RETRIEVAL_TASKS
        for metrics in result["evaluations"][task]["groups"].values()
    )
    for metrics in groups:
        intervals = metrics["confidence_intervals"]["difference_vs_pt_ml"]
        for name, bounds in intervals.items():
            if bounds != {"low": 0.0, "high": 0.0}:
                raise ExperimentValidationError(
                    f"Comparison 1 paired delta for {name} is not exactly zero: {bounds}"
                )


def precision_metric_delta(
    pre_quantization: Mapping[str, Any], post_quantization: Mapping[str, Any]
) -> dict[str, Any]:
    """Point-estimate changes caused by deployment quantization."""

    evaluation.validate_precision_pair(pre_quantization, post_quantization)
    pre_eval = pre_quantization["evaluations"]
    post_eval = post_quantization["evaluations"]
    english = {
        name: float(post_eval["english_voicechat_space"][name])
        - float(pre_eval["english_voicechat_space"][name])
        for name in ("r2", "cosine_mean", "cosine_p05")
    }
    retrieval: dict[str, Any] = {}
    for task in evaluation.RETRIEVAL_TASKS:
        retrieval[task] = {}
        for group, pre_metrics in pre_eval[task]["groups"].items():
            post_metrics = post_eval[task]["groups"][group]
            retrieval[task][group] = {
                metric: float(post_metrics[metric]) - float(pre_metrics[metric])
                for metric in evaluation.RETRIEVAL_METRICS
            } | {"hit_count": int(post_metrics["hit_count"]) - int(pre_metrics["hit_count"])}
    return {
        "direction": "post_quantization - pre_quantization",
        "english_voicechat_space": english,
        "retrieval": retrieval,
    }
