"""One evaluator and one result contract for comparisons 1--5.

All confidence intervals resample matched evaluation units with one set of
indices for the candidate and PT_ML reference.  Consequently the reported
``difference_vs_pt_ml`` intervals are paired, rather than differences between
two unrelated marginal intervals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .experiments import ExperimentValidationError, LAMBDAS

RESULT_SCHEMA_VERSION = "1.0"
PRECISION_STAGES = ("pre_quantization", "post_quantization")
RETRIEVAL_TASKS = (
    "candidate_on_english_retrieval",
    "historical_centered_fleurs_retrieval",
    "intrinsic_candidate_crosslingual_retrieval",
)
RETRIEVAL_METRICS = ("top1", "top5", "mrr", "median_rank")


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ExperimentValidationError(f"{name} must be a non-empty 2-D array")
    if not np.isfinite(result).all():
        raise ExperimentValidationError(f"{name} contains NaN or infinity")
    return result


def _same_shape(left: np.ndarray, right: np.ndarray, names: str) -> None:
    if left.shape != right.shape:
        raise ExperimentValidationError(
            f"{names} shapes differ: {left.shape} against {right.shape}; broadcasting is forbidden"
        )


def _percentile(values: np.ndarray, level: float) -> dict[str, float]:
    tail = (1.0 - level) / 2.0
    return {
        "low": float(np.quantile(values, tail)),
        "high": float(np.quantile(values, 1.0 - tail)),
    }


def _retrieval_summary(ranks: np.ndarray) -> dict[str, float | int]:
    ranks = np.asarray(ranks)
    return {
        "top1": float(np.mean(ranks == 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "hit_count": int(np.sum(ranks == 1)),
        "n": int(ranks.shape[0]),
    }


def retrieval_ranks(
    probe: np.ndarray, reference: np.ndarray, *, centered: bool = True
) -> np.ndarray:
    """Rank each row's same-index reference by cosine similarity."""

    probe = _matrix(probe, "probe")
    reference = _matrix(reference, "reference")
    _same_shape(probe, reference, "probe/reference")
    if centered:
        probe = probe - probe.mean(axis=0, keepdims=True)
        reference = reference - reference.mean(axis=0, keepdims=True)
    probe_norm = np.linalg.norm(probe, axis=1, keepdims=True)
    reference_norm = np.linalg.norm(reference, axis=1, keepdims=True)
    if np.any(probe_norm <= 1e-12) or np.any(reference_norm <= 1e-12):
        raise ExperimentValidationError("retrieval has a zero-norm embedding after centering")
    similarity = (probe / probe_norm) @ (reference / reference_norm).T
    truth = similarity.diagonal()[:, None]
    return (similarity > truth).sum(axis=1).astype(np.int64) + 1


def retrieval_metrics(
    probe: np.ndarray,
    reference: np.ndarray,
    *,
    pt_ml_probe: np.ndarray | None = None,
    centered: bool = True,
    confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Retrieval point estimates, absolute CIs, and paired deltas vs PT_ML."""

    if bootstrap_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ExperimentValidationError("invalid bootstrap_samples or confidence")
    ranks = retrieval_ranks(probe, reference, centered=centered)
    # Comparison 1 is its own reference and therefore has exact zero paired
    # differences.  Later comparisons pass the frozen PT_ML embeddings.
    baseline_ranks = (
        ranks.copy()
        if pt_ml_probe is None
        else retrieval_ranks(pt_ml_probe, reference, centered=centered)
    )
    if baseline_ranks.shape != ranks.shape:
        raise ExperimentValidationError("candidate and PT_ML retrieval query counts differ")

    rng = np.random.default_rng(seed)
    n = ranks.shape[0]
    sampled = rng.integers(0, n, size=(bootstrap_samples, n))
    candidate_samples = ranks[sampled]
    baseline_samples = baseline_ranks[sampled]

    def series(values: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "top1": np.mean(values == 1, axis=1),
            "top5": np.mean(values <= 5, axis=1),
            "mrr": np.mean(1.0 / values, axis=1),
            "median_rank": np.median(values, axis=1),
        }

    candidate_series = series(candidate_samples)
    baseline_series = series(baseline_samples)
    intervals = {
        "level": confidence,
        "method": "paired bootstrap over queries",
        "samples": bootstrap_samples,
        "seed": seed,
        "absolute": {
            metric: _percentile(candidate_series[metric], confidence)
            for metric in RETRIEVAL_METRICS
        },
        "difference_vs_pt_ml": {
            metric: _percentile(
                candidate_series[metric] - baseline_series[metric], confidence
            )
            for metric in RETRIEVAL_METRICS
        },
    }
    return {
        **_retrieval_summary(ranks),
        "centered": centered,
        "confidence_intervals": intervals,
    }


def embedding_diagnostics(embeddings: np.ndarray) -> dict[str, Any]:
    """Compact mean and per-embedding norm diagnostics without dumping vectors."""

    values = _matrix(embeddings, "embeddings")
    component_mean = values.mean(axis=0)
    norms = np.linalg.norm(values, axis=1)
    return {
        "n": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "finite": True,
        "global_mean": float(values.mean()),
        "global_std": float(values.std()),
        "mean_vector_l2": float(np.linalg.norm(component_mean)),
        "norm": {
            "mean": float(norms.mean()),
            "std": float(norms.std()),
            "min": float(norms.min()),
            "p05": float(np.quantile(norms, 0.05)),
            "p95": float(np.quantile(norms, 0.95)),
            "max": float(norms.max()),
        },
    }


def _voicechat_series(
    prediction: np.ndarray,
    target: np.ndarray,
    sampled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    error = np.square(prediction - target).sum(axis=1)
    centered_target = np.square(target - target.mean(axis=0, keepdims=True)).sum(axis=1)
    cosine = np.sum(prediction * target, axis=1) / np.clip(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1), 1e-12, None
    )
    r2 = 1.0 - error[sampled].sum(axis=1) / np.clip(
        centered_target[sampled].sum(axis=1), 1e-12, None
    )
    return r2, cosine[sampled].mean(axis=1)


def voicechat_space_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    pt_ml_prediction: np.ndarray | None = None,
    confidence: float = 0.95,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """English VoiceChat-space R2/cosine with paired intervals vs PT_ML."""

    prediction = _matrix(prediction, "VoiceChat prediction")
    target = _matrix(target, "FT_EN VoiceChat target")
    _same_shape(prediction, target, "VoiceChat prediction/target")
    baseline = prediction if pt_ml_prediction is None else _matrix(
        pt_ml_prediction, "PT_ML VoiceChat prediction"
    )
    _same_shape(baseline, target, "PT_ML prediction/target")
    if bootstrap_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ExperimentValidationError("invalid bootstrap_samples or confidence")

    error = prediction - target
    sse = float(np.square(error).sum())
    sst = float(np.square(target - target.mean(axis=0, keepdims=True)).sum())
    cosine = np.sum(prediction * target, axis=1) / np.clip(
        np.linalg.norm(prediction, axis=1) * np.linalg.norm(target, axis=1), 1e-12, None
    )
    rng = np.random.default_rng(seed)
    n = prediction.shape[0]
    # Work in modest chunks so a long frame-level evaluation does not allocate
    # bootstrap_samples * n indices all at once.
    candidate_r2, candidate_cosine = [], []
    baseline_r2, baseline_cosine = [], []
    remaining = bootstrap_samples
    while remaining:
        count = min(32, remaining)
        sampled = rng.integers(0, n, size=(count, n))
        one_r2, one_cosine = _voicechat_series(prediction, target, sampled)
        base_r2, base_cosine = _voicechat_series(baseline, target, sampled)
        candidate_r2.append(one_r2)
        candidate_cosine.append(one_cosine)
        baseline_r2.append(base_r2)
        baseline_cosine.append(base_cosine)
        remaining -= count
    candidate_r2_array = np.concatenate(candidate_r2)
    candidate_cosine_array = np.concatenate(candidate_cosine)
    baseline_r2_array = np.concatenate(baseline_r2)
    baseline_cosine_array = np.concatenate(baseline_cosine)
    return {
        "r2": 1.0 - sse / max(sst, 1e-12),
        "cosine_mean": float(cosine.mean()),
        "cosine_p05": float(np.quantile(cosine, 0.05)),
        "n": int(n),
        "confidence_intervals": {
            "level": confidence,
            "method": "paired bootstrap over frames",
            "samples": bootstrap_samples,
            "seed": seed,
            "absolute": {
                "r2": _percentile(candidate_r2_array, confidence),
                "cosine_mean": _percentile(candidate_cosine_array, confidence),
            },
            "difference_vs_pt_ml": {
                "r2": _percentile(candidate_r2_array - baseline_r2_array, confidence),
                "cosine_mean": _percentile(
                    candidate_cosine_array - baseline_cosine_array, confidence
                ),
            },
        },
    }


def evaluate_candidate(
    *,
    comparison: int,
    candidate_id: str,
    weight: float,
    precision: str,
    english_prediction: np.ndarray,
    english_target: np.ndarray,
    pt_ml_english_prediction: np.ndarray,
    retrieval_inputs: Mapping[
        str, Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
    ],
    diagnostic_embeddings: Mapping[str, np.ndarray],
    manifest_hashes: Mapping[str, str],
    seed: int = 0,
) -> dict[str, Any]:
    """Run every shared metric and return the versioned comparison record.

    Each retrieval tuple is ``(candidate_probe, reference, pt_ml_probe)`` and
    groups are typically ``overall`` for English or locale names for FLEURS.
    """

    if comparison not in range(1, 6):
        raise ExperimentValidationError("comparison must be between 1 and 5")
    if float(weight) not in LAMBDAS:
        raise ExperimentValidationError(f"lambda={weight} is outside the frozen sweep")
    if precision not in PRECISION_STAGES:
        raise ExperimentValidationError(f"precision must be one of {PRECISION_STAGES}")
    if set(manifest_hashes) != {"librispeech", "fleurs"}:
        raise ExperimentValidationError("manifest_hashes must contain exactly librispeech and fleurs")
    if set(retrieval_inputs) != set(RETRIEVAL_TASKS):
        raise ExperimentValidationError(
            f"retrieval inputs must contain exactly {RETRIEVAL_TASKS}"
        )
    evaluations: dict[str, Any] = {
        "english_voicechat_space": voicechat_space_metrics(
            english_prediction,
            english_target,
            pt_ml_prediction=pt_ml_english_prediction,
            seed=seed,
        )
    }
    for task in RETRIEVAL_TASKS:
        groups = retrieval_inputs[task]
        if not groups:
            raise ExperimentValidationError(f"{task} has no groups")
        evaluations[task] = {
            "groups": {
                group: retrieval_metrics(
                    candidate_probe,
                    reference,
                    pt_ml_probe=pt_ml_probe,
                    centered=True,
                    seed=seed,
                )
                for group, (candidate_probe, reference, pt_ml_probe) in groups.items()
            }
        }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "comparison": comparison,
        "candidate_id": candidate_id,
        "lambda": float(weight),
        "precision": precision,
        "manifests": dict(manifest_hashes),
        "selection_policy": {
            "map_fit": "LibriSpeech/map_train",
            "regularization_selection": "LibriSpeech/validation",
            "fleurs_tuning_allowed": False,
        },
        "evaluations": evaluations,
        "embedding_diagnostics": {
            name: embedding_diagnostics(values)
            for name, values in diagnostic_embeddings.items()
        },
    }
    validate_result(result)
    return result


def validate_result(result: Mapping[str, Any]) -> None:
    """Validate the stable result fields consumed by the final comparison."""

    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ExperimentValidationError("unsupported result schema")
    if result.get("precision") not in PRECISION_STAGES:
        raise ExperimentValidationError("result has invalid precision stage")
    evaluations = result.get("evaluations")
    expected = {"english_voicechat_space", *RETRIEVAL_TASKS}
    if not isinstance(evaluations, dict) or set(evaluations) != expected:
        raise ExperimentValidationError(f"result evaluations must be exactly {sorted(expected)}")
    english = evaluations["english_voicechat_space"]
    if not {"r2", "cosine_mean", "cosine_p05", "n", "confidence_intervals"} <= set(english):
        raise ExperimentValidationError("English VoiceChat-space result is incomplete")
    for task in RETRIEVAL_TASKS:
        groups = evaluations[task].get("groups")
        if not isinstance(groups, dict) or not groups:
            raise ExperimentValidationError(f"{task} groups are missing")
        for group, metrics in groups.items():
            required = {*RETRIEVAL_METRICS, "hit_count", "n", "confidence_intervals"}
            if not required <= set(metrics):
                raise ExperimentValidationError(f"{task}/{group} metrics are incomplete")
            intervals = metrics["confidence_intervals"]
            if "difference_vs_pt_ml" not in intervals:
                raise ExperimentValidationError(f"{task}/{group} lacks paired intervals")
    if not result.get("embedding_diagnostics"):
        raise ExperimentValidationError("result has no embedding mean/norm diagnostics")


def validate_precision_pair(
    pre_quantization: Mapping[str, Any], post_quantization: Mapping[str, Any]
) -> None:
    """Require comparable pre/post records for a deployment artifact."""

    validate_result(pre_quantization)
    validate_result(post_quantization)
    if pre_quantization["precision"] != "pre_quantization":
        raise ExperimentValidationError("first result is not pre_quantization")
    if post_quantization["precision"] != "post_quantization":
        raise ExperimentValidationError("second result is not post_quantization")
    for field in ("comparison", "candidate_id", "lambda", "manifests"):
        if pre_quantization.get(field) != post_quantization.get(field):
            raise ExperimentValidationError(
                f"pre/post quantization records differ in {field}; they are not a valid pair"
            )


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    validate_result(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
