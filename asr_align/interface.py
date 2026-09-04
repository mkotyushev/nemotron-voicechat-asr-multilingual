"""The encoder-to-LLM map: the one place an unconstrained linear map is allowed.

Inside the encoder, a re-basing has to survive five LayerNorms per block, which
only a permutation does -- hence the transport plans in
:mod:`asr_align.transport`. After the last block there is no LayerNorm left:
`norm_out` has already run, and the next thing that happens is

    embedding = hidden @ proj.weight.T + proj.bias

a single 1024 -> 4480 linear. So any linear map `M` applied to the swapped-in
encoder's output folds into `proj` exactly:

    proj.weight <- proj.weight @ M          proj.bias <- proj.weight @ m + proj.bias

and the deployed graph does not change at all -- same tensors, same shapes, same
C++. That is what this module fits and what `convert_asr_to_mmproj.py --align`
writes.

Five maps, in order of how much they are allowed to assume:

    identity      what the deployment does today, and the number to beat
    permutation   an optimal-transport plan and nothing else
    diagonal      per-channel scale and shift
    orthogonal    Procrustes: a rotation, optionally with one global scale
    linear        ridge regression, 1024 x 1024

There is deliberately no sixth. Refitting `proj` outright, 1024 -> 4480, looks
more expressive and is not: ridge is linear in its target, and the embedding
target is `proj` applied to the hidden target, so for every penalty the 4480-wide
fit is exactly the 1024-wide fit composed with `proj`. Measured, the two agree to
1e-5 across the whole sweep. A 1024 x 1024 correction is the most a map that
folds into `proj` can be, which also means the ladder here is the whole ladder.

Expressiveness is not free. The regression target only exists for English --
VoiceChat's encoder is the thing being matched and it understands nothing else
-- so a map with a million parameters fitted on English can encode English-only
structure in a way a rotation cannot. `linear` is the most likely to answer the
test sentence and the least likely to carry to the other 39 language-locales.
The report prints every rung so that choice is made with numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Moments:
    """Streaming first and second moments of a paired (source, target) stream."""

    n_source: int
    n_target: int
    count: int = 0
    sxx: torch.Tensor = field(init=False)
    sxy: torch.Tensor = field(init=False)
    sx: torch.Tensor = field(init=False)
    sy: torch.Tensor = field(init=False)
    syy: float = 0.0

    def __post_init__(self) -> None:
        self.sxx = torch.zeros(self.n_source, self.n_source, dtype=torch.float64)
        self.sxy = torch.zeros(self.n_source, self.n_target, dtype=torch.float64)
        self.sx = torch.zeros(self.n_source, dtype=torch.float64)
        self.sy = torch.zeros(self.n_target, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """``x`` and ``y`` are ``(frames, units)`` and share their frame axis."""

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{x.shape[0]} source frames against {y.shape[0]} target frames")
        x = x.float()
        y = y.float()
        self.sxx += (x.t() @ x).double().cpu()
        self.sxy += (x.t() @ y).double().cpu()
        self.sx += x.sum(dim=0).double().cpu()
        self.sy += y.sum(dim=0).double().cpu()
        self.syy += float(y.square().sum())
        self.count += int(x.shape[0])

    def centered(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_x = self.sx / self.count
        mean_y = self.sy / self.count
        cxx = self.sxx / self.count - torch.outer(mean_x, mean_x)
        cxy = self.sxy / self.count - torch.outer(mean_x, mean_y)
        return cxx, cxy, mean_x, mean_y


@dataclass(frozen=True)
class AffineMap:
    """``y = x @ weight + bias``, in the row convention the moments use."""

    weight: torch.Tensor
    bias: torch.Tensor
    name: str
    detail: dict[str, Any] = field(default_factory=dict)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x.double() @ self.weight + self.bias

    def fold_into_projection(
        self, proj_weight: torch.Tensor, proj_bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose with VoiceChat's `proj`, giving the tensors the mmproj holds.

        `proj_weight` is `(4480, 1024)` and `proj_bias` is `(4480,)`, the shapes
        `nn.Linear` and the GGUF both use. When this map already targets the
        4480-wide embedding, `proj` has been folded in during the fit and there
        is nothing left to compose with.
        """

        weight = self.weight.double()
        bias = self.bias.double()
        if weight.shape[1] == proj_weight.shape[0]:
            return weight.t().contiguous().float(), bias.float()
        if weight.shape[1] != proj_weight.shape[1]:
            raise ValueError(
                f"map {self.name} produces {weight.shape[1]} units, which is "
                f"neither the encoder width {proj_weight.shape[1]} nor the "
                f"projection width {proj_weight.shape[0]}"
            )
        p = proj_weight.double()
        return (p @ weight.t()).contiguous().float(), (p @ bias + proj_bias.double()).float()


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def identity_map(moments: Moments) -> AffineMap:
    """Feed the new encoder's output to the old `proj` untouched.

    This is the deployment as it stands, and every other row of the report is
    only interesting relative to it.
    """

    if moments.n_source != moments.n_target:
        raise ValueError("the identity map needs equal widths")
    return AffineMap(
        torch.eye(moments.n_source, dtype=torch.float64),
        torch.zeros(moments.n_target, dtype=torch.float64),
        "identity",
    )


def permutation_map(moments: Moments, permutation: torch.Tensor) -> AffineMap:
    """An optimal-transport plan, and nothing else.

    ``permutation[j]`` is the source unit target unit ``j`` draws from, which is
    what :meth:`TransportOperators.permutation` returns.
    """

    weight = torch.zeros(moments.n_source, moments.n_target, dtype=torch.float64)
    weight[permutation, torch.arange(moments.n_target)] = 1.0
    return AffineMap(weight, torch.zeros(moments.n_target, dtype=torch.float64), "permutation")


def diagonal_map(moments: Moments) -> AffineMap:
    """Per-channel scale and shift: the least a map can do and still do something.

    Channel `j` gets the least-squares line through `(x_j, y_j)`, so this
    corrects for two encoders whose channels mean the same thing at different
    gains and offsets -- the drift a long continued-training run tends to
    produce -- and for nothing else.
    """

    if moments.n_source != moments.n_target:
        raise ValueError("the diagonal map needs equal widths")
    cxx, cxy, mean_x, mean_y = moments.centered()
    variance = cxx.diagonal().clamp_min(1e-12)
    scale = cxy.diagonal() / variance
    return AffineMap(torch.diag(scale), mean_y - scale * mean_x, "diagonal")


def orthogonal_map(moments: Moments, *, scaled: bool = True) -> AffineMap:
    """Procrustes: the rotation that best takes one encoder's output to the other's.

    A rotation cannot invent or destroy structure, only re-express it, so of the
    non-trivial rungs this is the one whose English fit says the most about the
    other languages. `scaled` adds the single global factor that makes it a
    similarity rather than an isometry.
    """

    if moments.n_source != moments.n_target:
        raise ValueError("the orthogonal map needs equal widths")
    cxx, cxy, mean_x, mean_y = moments.centered()
    u, singular, vh = torch.linalg.svd(cxy)
    rotation = u @ vh
    factor = 1.0
    if scaled:
        factor = float(singular.sum() / cxx.diagonal().sum().clamp_min(1e-12))
    weight = rotation * factor
    return AffineMap(
        weight,
        mean_y - mean_x @ weight,
        "orthogonal",
        {"scale": factor, "explained_singular_mass": float(singular.sum())},
    )


def ridge_map(moments: Moments, alpha: float, name: str = "linear") -> AffineMap:
    """Ridge regression from the source encoder onto the target.

    ``alpha`` is relative: the penalty added to the diagonal is ``alpha`` times
    the mean source variance, so the same value means the same thing whatever
    the activations' scale.
    """

    cxx, cxy, mean_x, mean_y = moments.centered()
    penalty = alpha * float(cxx.diagonal().mean())
    lhs = cxx + penalty * torch.eye(cxx.shape[0], dtype=cxx.dtype)
    weight = torch.linalg.solve(lhs, cxy)
    return AffineMap(weight, mean_y - mean_x @ weight, name, {"alpha": alpha})


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def score(
    mapping: AffineMap,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    projection: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    """How well ``mapping`` reproduces the target on held-out frames.

    ``y`` should be the target the language model actually reads -- VoiceChat's
    4480-wide embedding -- and ``projection`` is `proj`, used to lift a map whose
    output is still 1024 wide. Scoring there rather than in the encoder's own
    space is the point: a direction the two encoders disagree on that `proj`
    discards costs nothing, and one it amplifies costs more than its size in the
    encoder suggests.
    """

    predicted = mapping.apply(x)
    target = y.double()
    if predicted.shape[1] != target.shape[1]:  # noqa: E501
        if projection is None:
            raise ValueError(
                f"map {mapping.name} produces {predicted.shape[1]} units against a "
                f"{target.shape[1]}-wide target, and no projection was given"
            )
        weight, bias = projection
        predicted = predicted @ weight.double().t() + bias.double()
    if predicted.shape[1] != target.shape[1]:
        raise ValueError(
            f"map {mapping.name} still produces {predicted.shape[1]} units after "
            f"the projection, against a {target.shape[1]}-wide target"
        )
    return metrics(predicted, target)


def metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = predicted - target
    sse = float(error.square().sum())
    sst = float((target - target.mean(dim=0, keepdim=True)).square().sum())
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=1)
    return {
        "r2": 1.0 - sse / max(sst, 1e-12),
        "relative_error": float(np.sqrt(sse / max(float(target.square().sum()), 1e-12))),
        "cosine_mean": float(cosine.mean()),
        "cosine_p05": float(cosine.quantile(0.05)),
    }


@torch.no_grad()
def score_deployed(
    mapping: AffineMap,
    x: torch.Tensor,
    y: torch.Tensor,
    projection: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """Score the map the way the server will run it, not the way it was fitted.

    What ships is not this map -- it is the map folded into `proj` and then
    quantized to Q8_0 by the converter. That matters more here than it would for
    a trained projection: a ridge fit is free to put large, cancelling weights
    wherever the calibration data does not object, and the runtime pays for that
    twice, once in the weights' own 8 bits and again in the 8 bits it quantizes
    the activations to. Measured on the test clip, a trained `proj` reproduces
    the runtime to 0.02 of a standard deviation and a fitted one to 0.07.

    So the penalty that looks best in float64 is not necessarily the one that
    serves best, and `--map auto` chooses on this rather than on the fit.
    """

    from .weights import q8_0  # local: weights imports nothing from here

    weight, bias = mapping.fold_into_projection(*projection)
    predicted = x.double() @ q8_0(weight).double().t() + bias.double()
    return metrics(predicted, y.double())


def select_alpha(
    moments: Moments,
    x: torch.Tensor,
    y: torch.Tensor,
    alphas: tuple[float, ...],
    *,
    projection: tuple[torch.Tensor, torch.Tensor] | None = None,
    name: str = "linear",
) -> tuple[AffineMap, list[dict[str, Any]]]:
    """Pick the ridge penalty on held-out speakers, and show the whole sweep.

    The sweep is worth printing rather than discarding: a fit that is flat in
    `alpha` is one whose 1024 x 1024 has found real structure, and one that
    needs a small `alpha` to look good is one that is memorizing the calibration
    speakers -- which is the failure mode that would keep the other languages
    broken.
    """

    trace: list[dict[str, Any]] = []
    best: AffineMap | None = None
    best_r2 = -float("inf")
    for alpha in alphas:
        candidate = ridge_map(moments, alpha, name)
        fitted = score(candidate, x, y, projection=projection)
        row = {"alpha": alpha, **fitted}
        if projection is not None:
            deployed = score_deployed(candidate, x, y, projection)
            row["deployed_r2"] = deployed["r2"]
            row["deployed_cosine_mean"] = deployed["cosine_mean"]
        # Choose on what the server will run when that is knowable.
        criterion = row.get("deployed_r2", row["r2"])
        trace.append(row)
        if criterion > best_r2:
            best, best_r2 = candidate, criterion
    assert best is not None
    return best, trace


def score_deployed_row(
    mapping: AffineMap,
    x: torch.Tensor,
    y: torch.Tensor,
    projection: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """:func:`score_deployed`, named for merging into a report row."""

    deployed = score_deployed(mapping, x, y, projection)
    return {"deployed_r2": deployed["r2"], "deployed_cosine_mean": deployed["cosine_mean"]}
