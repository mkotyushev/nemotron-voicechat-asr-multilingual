"""Comparison 3: learn the final activation correspondence and nothing else.

The encoder is left exactly as ``PT_ML`` shipped it.  The only thing fitted is
the correspondence between the two pretrained encoders' final activations,

    h_E A_L ~= h_M          h_M B_L ~= h_E

and only the reverse direction ever reaches the deployment, because after the
last block there is no LayerNorm left and ``proj`` is the next operation, so a
linear map on the encoder output folds into it exactly::

    W_proj,M = W_proj,F B_L^T        b_proj,M = W_proj,F b_L + b_proj,F

That makes this arm the interface-alignment control for comparisons 4 and 5: it
moves the same interface those arms move, without touching a single encoder
tensor, so whatever it buys is the part of their result that transport of the
fine-tuning delta does not explain.

Two things differ from :mod:`asr_align.interface`, which fits the same shape of
object for the exploratory CLI.  Both maps are regularized toward the identity
rather than toward zero::

    min_W  ||X W - Y||^2 + p ||W - I||^2     p = alpha * mean(diag(Cxx))

so ``alpha -> inf`` sends the linear part toward identity. The unpenalized
centering offset remains ``mean_Y - mean_X``; this limit is a mean-shift map,
not the untouched affine interface used by Comparison 1.
And regularization is selected on held-out LibriSpeech speakers by the map's own
target-space R2, not by the VoiceChat-space R2 the shared evaluator reports:
the frozen validation split is also the evaluation split, so selecting on the
headline metric would be selecting on the number being reported.

The expensive audio passes live in :mod:`final_map_projection`.  Everything
here is cheap and independently testable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from . import evaluation
from .experiments import ExperimentValidationError, sha256_file
from .interface import AffineMap, Moments

COMPARISON = 3
ARTIFACT_KIND = "final_activation_map"
CANDIDATE_ID = "final-map-projection"
# The shared evaluator keys every record by the frozen sweep; this arm mixes no
# task vector at all, which is that sweep's lambda=0.
CANDIDATE_LAMBDA = 0.0

# Identity-regularization strengths, relative to the mean source variance so the
# same value means the same thing whatever the activations' scale. This spans
# weak through strong regularization of the linear part; the separate identity
# diagnostic supplies the untouched Comparison 1 interface, including zero bias.
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)

FORWARD = "forward"
REVERSE = "reverse"
DIRECTIONS = {
    FORWARD: {
        "symbol": "A_L",
        "formula": "h_E A_L ~= h_M",
        "source": "E/PT_EN",
        "target": "M/PT_ML",
        "deployed": False,
    },
    REVERSE: {
        "symbol": "B_L",
        "formula": "h_M B_L ~= h_E",
        "source": "M/PT_ML",
        "target": "E/PT_EN",
        "deployed": True,
    },
}

# The fold is exact in exact arithmetic; what is measured here is the F32
# rounding of the composed projection the artifact carries.
FOLD_RELATIVE_TOLERANCE = 1e-4
SCORE_CHUNK_FRAMES = 4096


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def identity_ridge_map(moments: Moments, alpha: float, name: str) -> AffineMap:
    """Ridge regression pulled toward the identity instead of toward zero.

    The closed form of ``min ||X W - Y||^2 + p ||W - I||^2`` differs from plain
    ridge only in the right-hand side::

        W = (Cxx + p I)^-1 (Cxy + p I)

    and the bias is the centering offset the training means imply, so an
    ``alpha`` large enough to swamp ``Cxx`` leaves a map that shifts the mean and
    otherwise does nothing.
    """

    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ExperimentValidationError(f"identity-ridge alpha must be finite and >= 0, not {alpha}")
    if moments.n_source != moments.n_target:
        raise ExperimentValidationError(
            "the final activation map is square by construction; "
            f"got {moments.n_source} -> {moments.n_target}"
        )
    if moments.count <= 0:
        raise ExperimentValidationError("no frames were accumulated for the final activation map")
    cxx, cxy, mean_x, mean_y = moments.centered()
    penalty = alpha * float(cxx.diagonal().mean())
    eye = torch.eye(cxx.shape[0], dtype=cxx.dtype)
    weight = torch.linalg.solve(cxx + penalty * eye, cxy + penalty * eye)
    if not bool(torch.isfinite(weight).all()):
        raise ExperimentValidationError(f"identity-ridge map {name} at alpha={alpha:g} is not finite")
    bias = mean_y - mean_x @ weight
    return AffineMap(
        weight,
        bias,
        name,
        {
            "alpha": alpha,
            "penalty": penalty,
            "regularized_toward": "identity",
            "fit_frames": int(moments.count),
            "source_mean_l2": float(torch.linalg.vector_norm(mean_x)),
            "target_mean_l2": float(torch.linalg.vector_norm(mean_y)),
            "centering": "training-set means",
        },
    )


def compose(first: AffineMap, second: AffineMap, name: str) -> AffineMap:
    """``x -> (x @ W1 + b1) @ W2 + b2`` as one affine map."""

    if first.weight.shape[1] != second.weight.shape[0]:
        raise ExperimentValidationError(
            f"cannot compose {first.name} ({tuple(first.weight.shape)}) with "
            f"{second.name} ({tuple(second.weight.shape)})"
        )
    weight = first.weight.double() @ second.weight.double()
    bias = first.bias.double() @ second.weight.double() + second.bias.double()
    return AffineMap(weight, bias, name, {"composed_from": [first.name, second.name]})


def identity_like(mapping: AffineMap, name: str = "identity") -> AffineMap:
    """The untouched interface, for scoring the map against doing nothing."""

    width = mapping.weight.shape[0]
    return AffineMap(
        torch.eye(width, dtype=torch.float64),
        torch.zeros(mapping.weight.shape[1], dtype=torch.float64),
        name,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class ScoreAccumulator:
    """Streaming R2 and cosine of one map, over frames that arrive in batches.

    FLEURS recordings have no common length and LibriSpeech held-out frames are
    numerous enough to be worth not materializing twice, so the sums are kept
    rather than the activations.  Per-frame cosines are scalars, so those are
    kept in full and the reported percentile is exact.
    """

    mapping: AffineMap
    count: int = 0
    sse: float = 0.0
    syy: float = 0.0
    sy: torch.Tensor | None = None
    cosines: list[torch.Tensor] = field(default_factory=list)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """``x`` and ``y`` are ``(frames, units)`` and share their frame axis."""

        if x.ndim != 2 or y.ndim != 2:
            raise ExperimentValidationError("map scoring needs (frames, units) matrices")
        if x.shape[0] != y.shape[0]:
            raise ExperimentValidationError(
                f"{x.shape[0]} source frames against {y.shape[0]} target frames"
            )
        # The maps are fitted in F64 on the CPU while the activations arrive
        # from whichever device the encoders ran on; scoring follows the map.
        device = self.mapping.weight.device
        predicted = self.mapping.apply(x.to(device=device, dtype=torch.float64))
        target = y.to(device=device, dtype=torch.float64)
        if predicted.shape != target.shape:
            raise ExperimentValidationError(
                f"map {self.mapping.name} produces {tuple(predicted.shape)} against a "
                f"{tuple(target.shape)} target; broadcasting is forbidden"
            )
        if not bool(torch.isfinite(predicted).all() and torch.isfinite(target).all()):
            raise ExperimentValidationError(f"map {self.mapping.name} scoring saw NaN or infinity")
        if self.sy is None:
            self.sy = torch.zeros(target.shape[1], dtype=torch.float64)
        self.sse += float((predicted - target).square().sum())
        self.syy += float(target.square().sum())
        self.sy += target.sum(dim=0)
        self.cosines.append(
            torch.nn.functional.cosine_similarity(predicted, target, dim=1).cpu()
        )
        self.count += int(target.shape[0])

    def result(self) -> dict[str, float | int]:
        if self.count == 0 or self.sy is None:
            raise ExperimentValidationError("map scoring saw no frames")
        sst = self.syy - float(self.sy.square().sum()) / self.count
        cosine = torch.cat(self.cosines)
        return {
            "r2": 1.0 - self.sse / max(sst, 1e-12),
            "relative_error": math.sqrt(self.sse / max(self.syy, 1e-12)),
            "cosine_mean": float(cosine.mean()),
            "cosine_p05": float(cosine.quantile(0.05)),
            "n_frames": self.count,
        }


def score_map(
    mapping: AffineMap,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    chunk: int = SCORE_CHUNK_FRAMES,
) -> dict[str, float | int]:
    """:class:`ScoreAccumulator` over one pair of already-collected matrices."""

    if chunk <= 0:
        raise ExperimentValidationError("scoring chunk must be positive")
    accumulator = ScoreAccumulator(mapping)
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        accumulator.update(x[start:stop], y[start:stop])
    return accumulator.result()


def conditioning(mapping: AffineMap) -> dict[str, Any]:
    """Singular values and the condition number of the map's linear part."""

    singular = torch.linalg.svdvals(mapping.weight.double())
    largest = float(singular[0])
    smallest = float(singular[-1])
    return {
        "condition_number": largest / max(smallest, 1e-30),
        "singular_values": {
            "count": int(singular.numel()),
            "max": largest,
            "p95": float(singular.quantile(0.95)),
            "median": float(singular.median()),
            "p05": float(singular.quantile(0.05)),
            "min": smallest,
            "sum": float(singular.sum()),
            # An effective rank in the plainest sense: how much of the spectrum
            # sits above a thousandth of the largest direction.
            "above_1e-3_of_max": int((singular > largest * 1e-3).sum()),
        },
        "values": [float(value) for value in singular],
    }


def identity_distance(mapping: AffineMap) -> dict[str, Any]:
    """How far the fitted map is from leaving the interface alone."""

    weight = mapping.weight.double()
    width = weight.shape[0]
    if weight.shape[0] != weight.shape[1]:
        raise ExperimentValidationError("identity distance needs a square map")
    difference = weight - torch.eye(width, dtype=torch.float64)
    off_diagonal = weight - torch.diag(weight.diagonal())
    return {
        "frobenius": float(torch.linalg.matrix_norm(difference)),
        # ||I||_F is sqrt(width), so this is the change as a fraction of doing
        # nothing at all.
        "relative_frobenius": float(torch.linalg.matrix_norm(difference)) / math.sqrt(width),
        "max_abs": float(difference.abs().max()),
        "diagonal_mean": float(weight.diagonal().mean()),
        "diagonal_min": float(weight.diagonal().min()),
        "off_diagonal_frobenius": float(torch.linalg.matrix_norm(off_diagonal)),
        "bias_l2": float(torch.linalg.vector_norm(mapping.bias.double())),
    }


def cycle_consistency(forward: AffineMap, reverse: AffineMap) -> dict[str, Any]:
    """Whether the two directions undo each other, as weights alone can say.

    This is the cheap half.  The empirical half -- pushing held-out activations
    through both maps -- is scored by the runner with :func:`score_map` on the
    composed map, because a round trip that is poor on paper may still be exact
    on the subspace the data actually occupies.
    """

    rows: dict[str, Any] = {}
    for name, first, second in (
        ("E_to_M_to_E", forward, reverse),
        ("M_to_E_to_M", reverse, forward),
    ):
        composed = compose(first, second, name)
        rows[name] = identity_distance(composed)
    return {
        "definition": "distance from identity of each composed round trip",
        "round_trips": rows,
    }


def map_report(
    mapping: AffineMap,
    *,
    direction: str,
    selection: Mapping[str, Any],
    held_out: Mapping[str, Any],
    fit: Mapping[str, Any],
) -> dict[str, Any]:
    """One direction's complete record, minus the FLEURS generalization test."""

    if direction not in DIRECTIONS:
        raise ExperimentValidationError(f"unknown map direction {direction!r}")
    return {
        **DIRECTIONS[direction],
        "direction": direction,
        "name": mapping.name,
        "shape": list(mapping.weight.shape),
        "alpha": float(mapping.detail["alpha"]),
        "penalty": float(mapping.detail["penalty"]),
        "regularized_toward": "identity",
        "centering": "training-set means",
        "fit": dict(fit),
        "selection": dict(selection),
        "held_out": dict(held_out),
        "conditioning": conditioning(mapping),
        "identity_distance": identity_distance(mapping),
    }


def select_alpha(
    moments: Moments,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    name: str,
    alphas: Sequence[float] = ALPHAS,
    chunk: int = SCORE_CHUNK_FRAMES,
) -> tuple[AffineMap, list[dict[str, Any]]]:
    """Fit every ``alpha`` and keep the one with the best held-out target R2.

    The whole sweep is returned so sensitivity to regularization can be
    inspected. Selecting a weak penalty alone does not establish overfitting
    or predict how the map will behave on other languages.
    """

    if not alphas:
        raise ExperimentValidationError("the identity-regularization sweep is empty")
    trace: list[dict[str, Any]] = []
    best: AffineMap | None = None
    best_r2 = -math.inf
    for alpha in alphas:
        candidate = identity_ridge_map(moments, alpha, name)
        scored = score_map(candidate, x, y, chunk=chunk)
        distance = identity_distance(candidate)
        singular = torch.linalg.svdvals(candidate.weight.double())
        trace.append(
            {
                "alpha": float(alpha),
                **scored,
                "condition_number": float(singular[0]) / max(float(singular[-1]), 1e-30),
                "identity_relative_frobenius": distance["relative_frobenius"],
            }
        )
        if scored["r2"] > best_r2:
            best, best_r2 = candidate, float(scored["r2"])
    if best is None:  # pragma: no cover - the sweep is non-empty above
        raise ExperimentValidationError("no identity-ridge map was selected")
    return best, trace


# ---------------------------------------------------------------------------
# Folding into the deployed projection
# ---------------------------------------------------------------------------


def fold_reverse_map(
    reverse: AffineMap, proj_weight: torch.Tensor, proj_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Compose ``B_L`` with the untouched FT_EN projection, offset included."""

    if proj_weight.ndim != 2 or proj_bias.ndim != 1:
        raise ExperimentValidationError("the VoiceChat projection must be (out, in) and (out,)")
    if proj_weight.shape[0] != proj_bias.shape[0]:
        raise ExperimentValidationError("projection weight and bias disagree on the output width")
    if tuple(reverse.weight.shape) != (proj_weight.shape[1], proj_weight.shape[1]):
        raise ExperimentValidationError(
            f"reverse map {tuple(reverse.weight.shape)} does not match the "
            f"{proj_weight.shape[1]}-wide encoder output"
        )
    weight, bias = reverse.fold_into_projection(proj_weight, proj_bias)
    if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
        raise ExperimentValidationError("the folded projection is not finite")
    if weight.shape != proj_weight.shape or bias.shape != proj_bias.shape:
        raise ExperimentValidationError("folding changed the deployed projection's shape")
    report = {
        "formula": {
            "weight": "W_proj,M = W_proj,F B_L^T",
            "bias": "b_proj,M = W_proj,F b_L + b_proj,F",
        },
        "encoder_tensors_changed": False,
        "graph_changed": False,
        "shape": {"weight": list(weight.shape), "bias": list(bias.shape)},
        "original_norm": {
            "weight_l2": float(torch.linalg.matrix_norm(proj_weight.double())),
            "bias_l2": float(torch.linalg.vector_norm(proj_bias.double())),
        },
        "folded_norm": {
            "weight_l2": float(torch.linalg.matrix_norm(weight.double())),
            "bias_l2": float(torch.linalg.vector_norm(bias.double())),
        },
    }
    report["norm_ratio"] = {
        "weight": report["folded_norm"]["weight_l2"]
        / max(report["original_norm"]["weight_l2"], 1e-30),
        "bias": report["folded_norm"]["bias_l2"]
        / max(report["original_norm"]["bias_l2"], 1e-30),
    }
    return weight, bias, report


def verify_folding(
    reverse: AffineMap,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    folded_weight: torch.Tensor,
    folded_bias: torch.Tensor,
    hidden: torch.Tensor,
    *,
    tolerance: float = FOLD_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Require the folded projection to reproduce map-then-project on real frames."""

    if hidden.ndim != 2:
        raise ExperimentValidationError("fold verification needs (frames, units) activations")
    mapped = reverse.apply(hidden.double())
    expected = mapped @ proj_weight.double().t() + proj_bias.double()
    observed = hidden.double() @ folded_weight.double().t() + folded_bias.double()
    if expected.shape != observed.shape:
        raise ExperimentValidationError("fold verification shapes differ; broadcasting is forbidden")
    difference = observed - expected
    relative = float(torch.linalg.matrix_norm(difference)) / max(
        float(torch.linalg.matrix_norm(expected)), 1e-30
    )
    report = {
        "frames": int(hidden.shape[0]),
        "max_abs": float(difference.abs().max()),
        "relative_l2": relative,
        "tolerance": float(tolerance),
        "passed": relative <= tolerance,
        "note": "difference is the F32 rounding of the composed projection",
    }
    if not report["passed"]:
        raise ExperimentValidationError(
            f"the folded projection does not reproduce the map: relative L2 {relative:g}"
        )
    return report


# ---------------------------------------------------------------------------
# The encoder must not move
# ---------------------------------------------------------------------------


def encoder_byte_digest(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """SHA-256 over the canonical encoder tensors' raw bytes, keys included.

    ``torch.equal`` treats ``-0.0`` and ``0.0`` as equal, so value equality is
    not quite the claim Comparison 3 makes about ``encoder.*``.  This hashes the
    bytes.
    """

    digest = hashlib.sha256()
    keys = sorted(key for key in state if key.startswith("encoder."))
    if not keys:
        raise ExperimentValidationError("no canonical encoder tensors to hash")
    values = 0
    for key in keys:
        value = state[key].detach().to(dtype=torch.float32).contiguous().cpu()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
        values += int(value.numel())
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": len(keys),
        "value_count": values,
        "dtype": "float32",
    }


def assert_encoder_byte_identical(
    expected: Mapping[str, torch.Tensor], observed: Mapping[str, torch.Tensor], *, label: str
) -> dict[str, Any]:
    """Reject any Comparison 3 artifact whose encoder is not ``PT_ML`` itself."""

    left = encoder_byte_digest(expected)
    right = encoder_byte_digest(observed)
    if left != right:
        raise ExperimentValidationError(
            f"{label}: encoder tensors are not byte-identical to PT_ML "
            f"({left['sha256']} against {right['sha256']})"
        )
    return {"byte_identical": True, "compared_with": label, **left}


# ---------------------------------------------------------------------------
# The frozen Comparison 1 activation cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivationCache:
    """The Comparison 1 residual-boundary cache, verified before it is read."""

    index_path: Path
    root: Path
    index: dict[str, Any]
    final_key: str
    splits: dict[str, list[dict[str, Any]]]

    def records(self, split: str) -> list[dict[str, Any]]:
        if split not in self.splits:
            raise ExperimentValidationError(f"the frozen activation cache has no {split!r} split")
        return self.splits[split]


def load_activation_cache(
    path: Path, *, manifest_sha256: str, n_layer: int, candidate_id: str
) -> ActivationCache:
    """Validate the Comparison 1 activation index and locate the final layer."""

    index_path = (path / "index.json") if path.is_dir() else path
    if not index_path.is_file():
        raise ExperimentValidationError(f"activation index does not exist: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("comparison") != 1 or index.get("candidate_id") != candidate_id:
        raise ExperimentValidationError("the activation cache is not the Comparison 1 PT_ML cache")
    if index.get("precision") != "pre_quantization":
        raise ExperimentValidationError("the activation cache is not a pre-quantization cache")
    if index.get("manifest_sha256") != manifest_sha256:
        raise ExperimentValidationError("the activation cache used a different LibriSpeech manifest")
    if index.get("reserved_test_encoded") is not False:
        raise ExperimentValidationError("the activation cache encoded the reserved test split")
    splits = index.get("splits")
    if not isinstance(splits, dict) or not {"map_train", "validation"} <= set(splits):
        raise ExperimentValidationError("the activation cache lacks map_train and validation")
    final_key = f"residual.block.{n_layer - 1}"
    for split, shards in splits.items():
        if not shards:
            raise ExperimentValidationError(f"the activation cache {split!r} split is empty")
        for shard in shards:
            tensors = shard.get("tensors", {})
            blocks = [name for name in tensors if name.startswith("residual.block.")]
            if len(blocks) != n_layer:
                raise ExperimentValidationError(
                    f"the activation cache holds {len(blocks)} block outputs, not {n_layer}"
                )
            if final_key not in tensors:
                raise ExperimentValidationError(
                    f"the activation cache has no final-layer tensor {final_key}"
                )
    return ActivationCache(
        index_path=index_path,
        root=index_path.parent.parent,
        index=index,
        final_key=final_key,
        splits={name: list(shards) for name, shards in splits.items()},
    )


def read_final_activations(
    cache: ActivationCache, shard: Mapping[str, Any]
) -> tuple[list[str], torch.Tensor]:
    """Return one shard's recordings and its verified final-layer activations."""

    from convert_asr_to_mmproj import SafeTensors  # local: keeps this module import-light

    path = (cache.root / str(shard["path"])).resolve()
    if not path.is_relative_to(cache.root):
        raise ExperimentValidationError("an activation shard path escapes the cache directory")
    if not path.is_file():
        raise ExperimentValidationError(f"activation shard is missing: {path}")
    if path.stat().st_size != int(shard.get("bytes", -1)):
        raise ExperimentValidationError(f"activation shard byte count changed: {path}")
    if sha256_file(path) != shard.get("sha256"):
        raise ExperimentValidationError(f"activation shard SHA-256 changed: {path}")
    source = SafeTensors(path)
    try:
        value = torch.from_numpy(source.f32(cache.final_key).copy())
    finally:
        source.f.close()
    expected_shape = [int(size) for size in shard["tensors"][cache.final_key]]
    if list(value.shape) != expected_shape:
        raise ExperimentValidationError(
            f"activation shard {path} holds {list(value.shape)}, not {expected_shape}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ExperimentValidationError(f"activation shard {path} contains NaN or infinity")
    records = [str(record) for record in shard["records"]]
    if len(records) != value.shape[0]:
        raise ExperimentValidationError(
            f"activation shard {path} covers {len(records)} recordings for {value.shape[0]} rows"
        )
    return records, value


def iter_shards(
    cache: ActivationCache, split: str
) -> Iterator[tuple[dict[str, Any], list[str], torch.Tensor]]:
    for shard in cache.records(split):
        records, value = read_final_activations(cache, shard)
        yield shard, records, value


# ---------------------------------------------------------------------------
# Comparison 1 is the only thing this arm is measured against
# ---------------------------------------------------------------------------


def comparison_delta_table(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Isolate interface alignment: Comparison 3 minus the untouched interface.

    The paired intervals come from the candidate record itself, where they were
    computed against the exact frozen PT_ML arrays; nothing is recomputed here
    from two unrelated marginal results.
    """

    evaluation.validate_result(candidate)
    evaluation.validate_result(baseline)
    if candidate.get("comparison") != COMPARISON or baseline.get("comparison") != 1:
        raise ExperimentValidationError("the delta table pairs Comparison 3 against Comparison 1")
    if candidate.get("precision") != baseline.get("precision"):
        raise ExperimentValidationError("the delta table needs one precision stage")
    if candidate.get("manifests") != baseline.get("manifests"):
        raise ExperimentValidationError("the delta table needs one set of frozen manifests")

    english_candidate = candidate["evaluations"]["english_voicechat_space"]
    english_baseline = baseline["evaluations"]["english_voicechat_space"]
    english = {
        metric: {
            "pt_ml": float(english_baseline[metric]),
            "final_map": float(english_candidate[metric]),
            "difference": float(english_candidate[metric]) - float(english_baseline[metric]),
        }
        for metric in ("r2", "cosine_mean", "cosine_p05")
    }
    for metric in ("r2", "cosine_mean"):
        english[metric]["paired_interval"] = english_candidate["confidence_intervals"][
            "difference_vs_pt_ml"
        ][metric]

    retrieval: dict[str, Any] = {}
    for task in evaluation.RETRIEVAL_TASKS:
        groups = candidate["evaluations"][task]["groups"]
        baseline_groups = baseline["evaluations"][task]["groups"]
        if set(groups) != set(baseline_groups):
            raise ExperimentValidationError(f"{task} groups differ between comparisons 1 and 3")
        rows: dict[str, Any] = {}
        for group, metrics in groups.items():
            reference = baseline_groups[group]
            row = {
                metric: {
                    "pt_ml": float(reference[metric]),
                    "final_map": float(metrics[metric]),
                    "difference": float(metrics[metric]) - float(reference[metric]),
                    "paired_interval": metrics["confidence_intervals"]["difference_vs_pt_ml"][
                        metric
                    ],
                }
                for metric in evaluation.RETRIEVAL_METRICS
            }
            row["hit_count"] = {
                "pt_ml": int(reference["hit_count"]),
                "final_map": int(metrics["hit_count"]),
                "difference": int(metrics["hit_count"]) - int(reference["hit_count"]),
            }
            row["n"] = int(metrics["n"])
            rows[group] = row
        retrieval[task] = rows
    return {
        "schema_version": "1.0",
        "comparison": COMPARISON,
        "measures": (
            "the effect of aligning the final activation interface alone, with the "
            "encoder byte-identical to PT_ML"
        ),
        "direction": "Comparison 3 minus Comparison 1",
        "precision": candidate["precision"],
        "manifests": dict(candidate["manifests"]),
        "english_voicechat_space": english,
        "retrieval": retrieval,
    }
