"""Strict, inexpensive helpers for Comparison 2 direct task arithmetic.

The expensive audio passes live in :mod:`direct_task_arithmetic`.  This module
keeps the task-vector accounting, frozen-baseline validation, activation-growth
tripwire, and Pareto-table construction independently testable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import baseline, evaluation
from .experiments import (
    LAMBDAS,
    ExperimentValidationError,
    canonical_encoder,
    sha256_file,
)

COMPARISON = 2
ARTIFACT_KIND = "direct_task_arithmetic"
TASK_VECTOR_ATOL = 1e-6
TASK_VECTOR_RTOL = 1e-6
MAX_ACTIVATION_GROWTH = 10.0


@dataclass(frozen=True)
class BaselineReference:
    """Validated Comparison 1 files consumed as the exact paired baseline."""

    root: Path
    run_path: Path
    run: dict[str, Any]
    embeddings_path: Path
    result_paths: dict[str, Path]
    results: dict[str, dict[str, Any]]


def lambda_label(weight: float) -> str:
    """Stable path-safe label for one member of the frozen sweep."""

    weight = float(weight)
    if weight not in LAMBDAS:
        raise ExperimentValidationError(f"lambda={weight:g} is outside the frozen sweep")
    return f"{weight:g}".replace(".", "p")


def candidate_id(weight: float) -> str:
    return f"direct-lambda-{lambda_label(weight)}"


def _tensor_groups(key: str) -> tuple[str, str]:
    parts = key.split(".")
    if len(parts) >= 4 and parts[1] == "layers":
        block = f"encoder.layers.{parts[2]}"
        suffix = ".".join(parts[3:])
        if suffix.startswith("self_attn."):
            module_type = "self_attention"
        elif suffix.startswith("feed_forward"):
            module_type = "feed_forward"
        elif suffix.startswith("conv."):
            module_type = "convolution"
        elif suffix.startswith("norm_"):
            module_type = "normalization"
        else:  # pragma: no cover - future architecture extension
            module_type = "other"
    elif key.startswith("encoder.subsampling."):
        block = "encoder.subsampling"
        module_type = "subsampling"
    else:  # canonical_encoder rejects an empty namespace, not unknown future keys
        block = "encoder.other"
        module_type = "other"
    return block, module_type


def _norm_row(value: torch.Tensor) -> dict[str, int | float]:
    double = value.detach().double()
    return {
        "tensor_count": 1,
        "value_count": int(double.numel()),
        "nonzero_count": int(torch.count_nonzero(double)),
        "l2": float(torch.linalg.vector_norm(double)),
        "rms": float(double.square().mean().sqrt()) if double.numel() else 0.0,
        "max_abs": float(double.abs().max()) if double.numel() else 0.0,
    }


def _aggregate(rows: Sequence[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, float | int]] = {}
    for group, value in rows:
        row = grouped.setdefault(
            group,
            {
                "tensor_count": 0,
                "value_count": 0,
                "nonzero_count": 0,
                "sum_squares": 0.0,
                "max_abs": 0.0,
            },
        )
        double = value.detach().double()
        row["tensor_count"] = int(row["tensor_count"]) + 1
        row["value_count"] = int(row["value_count"]) + double.numel()
        row["nonzero_count"] = int(row["nonzero_count"]) + int(
            torch.count_nonzero(double)
        )
        row["sum_squares"] = float(row["sum_squares"]) + float(double.square().sum())
        row["max_abs"] = max(float(row["max_abs"]), float(double.abs().max()))

    result: dict[str, Any] = {}
    for group, row in sorted(grouped.items()):
        sum_squares = float(row.pop("sum_squares"))
        value_count = int(row["value_count"])
        result[group] = {
            **row,
            "l2": math.sqrt(sum_squares),
            "rms": math.sqrt(sum_squares / max(value_count, 1)),
        }
    return result


def task_vector_report(
    e: Mapping[str, torch.Tensor],
    f: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    *,
    atol: float = TASK_VECTOR_ATOL,
    rtol: float = TASK_VECTOR_RTOL,
) -> dict[str, Any]:
    """Record task-vector norms and require ``E + (F - E) ~= F`` in F32."""

    ancestor = canonical_encoder(e, role="E/PT_EN")
    descendant = canonical_encoder(f, role="F/FT_EN")
    update = canonical_encoder(delta, role="delta_F")
    if set(ancestor) != set(descendant) or set(ancestor) != set(update):
        raise ExperimentValidationError("task-vector report tensor keys differ")

    tensor_rows: dict[str, Any] = {}
    block_values: list[tuple[str, torch.Tensor]] = []
    module_values: list[tuple[str, torch.Tensor]] = []
    total_values: list[tuple[str, torch.Tensor]] = []
    error_squared = 0.0
    target_squared = 0.0
    max_abs_error = 0.0
    exact_values = 0
    value_count = 0
    for key in sorted(update):
        if ancestor[key].shape != descendant[key].shape or ancestor[key].shape != update[key].shape:
            raise ExperimentValidationError(
                f"task-vector report shape mismatch for {key}; broadcasting is forbidden"
            )
        reconstructed = ancestor[key] + update[key]
        error = reconstructed.double() - descendant[key].double()
        tolerance = atol + rtol * descendant[key].double().abs()
        if not bool((error.abs() <= tolerance).all()):
            raise ExperimentValidationError(
                f"reconstruction invariant failed for {key}: "
                f"max_abs={float(error.abs().max()):g}"
            )
        block, module_type = _tensor_groups(key)
        norm = _norm_row(update[key])
        tensor_rows[key] = {
            "shape": list(update[key].shape),
            "dtype": str(update[key].dtype).removeprefix("torch."),
            "block": block,
            "module_type": module_type,
            **norm,
        }
        block_values.append((block, update[key]))
        module_values.append((module_type, update[key]))
        total_values.append(("encoder", update[key]))
        error_squared += float(error.square().sum())
        target_squared += float(descendant[key].double().square().sum())
        max_abs_error = max(max_abs_error, float(error.abs().max()))
        exact_values += int(torch.count_nonzero(error == 0))
        value_count += error.numel()

    total = _aggregate(total_values)["encoder"]
    return {
        "formula": "delta_F = F - E",
        "arithmetic_precision": "float32",
        "canonical_namespace": "encoder.",
        "reconstruction": {
            "formula": "E + delta_F ~= F",
            "passed": True,
            "atol": float(atol),
            "rtol": float(rtol),
            "max_abs_error": max_abs_error,
            "relative_l2_error": math.sqrt(error_squared)
            / max(math.sqrt(target_squared), 1e-30),
            "exact_value_count": exact_values,
            "value_count": value_count,
        },
        "total": total,
        "by_block": _aggregate(block_values),
        "by_module_type": _aggregate(module_values),
        "by_tensor": tensor_rows,
    }


def state_norm_report(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Compact encoder-weight norm used to screen candidate growth."""

    canonical = canonical_encoder(state, role="candidate")
    return _aggregate([("encoder", value) for value in canonical.values()])["encoder"]


def activation_profile(store: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Summarize one deterministic forward pass at every residual boundary."""

    if not store:
        raise ExperimentValidationError("forward profile contains no activations")
    result: dict[str, Any] = {}
    for name, value in sorted(store.items()):
        if not bool(torch.isfinite(value).all()):
            raise ExperimentValidationError(f"forward activation {name} is not finite")
        result[name] = {"shape": list(value.shape), **_norm_row(value)}
    return result


def activation_growth_report(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    maximum: float = MAX_ACTIVATION_GROWTH,
) -> dict[str, Any]:
    """Compare layerwise RMS/max norms to lambda=0 using a safety tripwire.

    The tripwire is a validity check, not a model-selection threshold: every
    candidate remains in the sweep unless its activations grow by an order of
    magnitude at a residual boundary.
    """

    if maximum <= 1.0:
        raise ExperimentValidationError("activation-growth maximum must be greater than one")
    if set(reference) != set(candidate):
        raise ExperimentValidationError("candidate residual-boundary hook points changed")
    rows: dict[str, Any] = {}
    observed_max = 0.0
    for name in sorted(reference):
        if reference[name].get("shape") != candidate[name].get("shape"):
            raise ExperimentValidationError(f"activation shape changed at {name}")
        ratios = {}
        for metric in ("rms", "max_abs"):
            denominator = max(float(reference[name][metric]), 1e-30)
            ratio = float(candidate[name][metric]) / denominator
            if not math.isfinite(ratio):
                raise ExperimentValidationError(f"activation growth at {name} is non-finite")
            ratios[f"{metric}_ratio_vs_lambda_0"] = ratio
            observed_max = max(observed_max, ratio)
        rows[name] = ratios
    if observed_max > maximum:
        raise ExperimentValidationError(
            f"abnormal activation growth: ratio {observed_max:g} exceeds {maximum:g}"
        )
    return {
        "passed": True,
        "policy": "residual-boundary RMS and max_abs must be <= limit times lambda=0",
        "limit": float(maximum),
        "maximum_observed_ratio": observed_max,
        "hook_points": rows,
    }


def _verified_record(root: Path, record: Mapping[str, Any], *, label: str) -> Path:
    relative = Path(str(record.get("path", "")))
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ExperimentValidationError(f"baseline {label} path escapes its output directory")
    if not path.is_file():
        raise ExperimentValidationError(f"baseline {label} is missing: {path}")
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ExperimentValidationError(f"baseline {label} byte count changed")
    if sha256_file(path) != record.get("sha256"):
        raise ExperimentValidationError(f"baseline {label} SHA-256 changed")
    return path


def load_baseline_reference(path: Path, shared: baseline.SharedSetup) -> BaselineReference:
    """Validate the completed Comparison 1 result and its frozen array cache."""

    root = path.resolve()
    run_path = root / "run.json" if root.is_dir() else root
    root = run_path.parent
    if not run_path.is_file():
        raise ExperimentValidationError(f"Comparison 1 run.json does not exist: {run_path}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run.get("comparison") != 1
        or run.get("candidate_id") != baseline.BASELINE_CANDIDATE_ID
        or run.get("status") != "complete"
    ):
        raise ExperimentValidationError("paired baseline is not a completed Comparison 1 run")
    recorded_setup = run.get("shared_setup", {})
    if recorded_setup.get("sha256") != shared.sha256:
        raise ExperimentValidationError("Comparison 1 used a different frozen shared setup")
    if recorded_setup.get("manifests") != shared.manifest_hashes:
        raise ExperimentValidationError("Comparison 1 used different frozen manifests")

    embeddings_path = _verified_record(
        root, run.get("reference_cache", {}).get("embeddings", {}), label="embedding cache"
    )
    result_paths: dict[str, Path] = {}
    results: dict[str, dict[str, Any]] = {}
    result_records = run.get("evaluation", {}).get("results", {})
    for stage in evaluation.PRECISION_STAGES:
        result_paths[stage] = _verified_record(
            root, result_records.get(stage, {}), label=f"{stage} result"
        )
        results[stage] = json.loads(result_paths[stage].read_text(encoding="utf-8"))
        evaluation.validate_result(results[stage])
        if results[stage].get("manifests") != shared.manifest_hashes:
            raise ExperimentValidationError(f"baseline {stage} result manifest hashes changed")
    evaluation.validate_precision_pair(
        results["pre_quantization"], results["post_quantization"]
    )
    return BaselineReference(
        root=root,
        run_path=run_path,
        run=run,
        embeddings_path=embeddings_path,
        result_paths=result_paths,
        results=results,
    )


def pareto_table(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tabulate English R2 against macro intrinsic cross-lingual MRR."""

    by_weight = {float(result["lambda"]): result for result in results}
    if set(by_weight) != set(LAMBDAS):
        raise ExperimentValidationError("Pareto table requires every frozen lambda exactly once")
    rows: list[dict[str, Any]] = []
    for weight in LAMBDAS:
        result = by_weight[weight]
        if result.get("precision") != "pre_quantization":
            raise ExperimentValidationError("Pareto sweep must use pre-quantization results")
        english = result["evaluations"]["english_voicechat_space"]
        intrinsic = result["evaluations"]["intrinsic_candidate_crosslingual_retrieval"][
            "groups"
        ]
        historical = result["evaluations"]["historical_centered_fleurs_retrieval"][
            "groups"
        ]
        rows.append(
            {
                "lambda": weight,
                "candidate_id": result["candidate_id"],
                "english_voicechat_r2": float(english["r2"]),
                "english_voicechat_cosine": float(english["cosine_mean"]),
                "multilingual_intrinsic_mrr_macro": sum(
                    float(metrics["mrr"]) for metrics in intrinsic.values()
                )
                / len(intrinsic),
                "multilingual_intrinsic_top1_macro": sum(
                    float(metrics["top1"]) for metrics in intrinsic.values()
                )
                / len(intrinsic),
                "voicechat_crosslingual_mrr_macro": sum(
                    float(metrics["mrr"]) for metrics in historical.values()
                )
                / len(historical),
            }
        )
    for row in rows:
        row["pareto_efficient"] = not any(
            other["english_voicechat_r2"] >= row["english_voicechat_r2"]
            and other["multilingual_intrinsic_mrr_macro"]
            >= row["multilingual_intrinsic_mrr_macro"]
            and (
                other["english_voicechat_r2"] > row["english_voicechat_r2"]
                or other["multilingual_intrinsic_mrr_macro"]
                > row["multilingual_intrinsic_mrr_macro"]
            )
            for other in rows
        )
    return {
        "schema_version": "1.0",
        "comparison": COMPARISON,
        "selection_performed": False,
        "english_transfer_axis": "English VoiceChat-space R2 against FT_EN",
        "multilingual_retention_axis": (
            "macro mean intrinsic candidate-to-candidate cross-lingual MRR"
        ),
        "rows": rows,
    }
