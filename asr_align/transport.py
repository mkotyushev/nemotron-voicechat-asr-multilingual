"""The optimal-transport core, adapted from `chmv2_distill/ot_width.py`.

Singh and Jaggi, "Model Fusion via Optimal Transport" (NeurIPS 2020), align two
networks by treating each layer's neurons as an unordered point cloud: run both
models on the same inputs, measure how alike each pair of neurons behaves, and
solve a transport problem between the two neuron distributions. The plan `T`
(`n_src` by `n_tgt`, marginals `mu` and `nu`) yields two operators:

``analysis = T / nu``
    columns sum to one, so it maps an *output* axis -- each target unit becomes
    a weighted average of source units;
``synthesis = T / mu``
    rows sum to one, so it maps an *input* axis, and `W = A.T @ W_src @ S`
    reproduces `W_src` exactly whenever `synthesis @ analysis.T` is the identity.

What is different here from the width-transfer case that module was written for:
**the two encoders are the same size**. Twenty-four layers, d_model 1024, eight
heads, ffn 4096, on both sides. With equal sizes and uniform marginals the exact
solver returns a permutation, so OT can only say *which neuron is which* -- it
cannot re-mix them. That is the right constraint to have inside the encoder,
because a permutation is the only re-basing that survives a LayerNorm, and there
are five of them per block. It is also, most likely, close to the identity: the
multilingual model is a continued-training descendant of the English one that
VoiceChat was fine-tuned from, and continued training moves weights without
renumbering neurons. :func:`identity_agreement` is how that gets checked rather
than assumed, and if it holds then the plans are a *negative* result worth
having -- it says the mismatch is not a permutation and sends the work to the
one place an unconstrained linear map is allowed, the encoder-to-LLM interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

GROUND_METRICS = {"cosine", "correlation"}
COLUMN_ASSIGNMENTS = {"barycentric", "integral"}
_EPSILON = 1e-12


def _require_pot():
    try:
        import ot
    except ImportError as error:  # pragma: no cover - exercised by env checks
        raise ImportError(
            "The barycentric plan needs the POT package: `uv pip install pot`, "
            "or pass --column-assignment integral."
        ) from error
    return ot


# ---------------------------------------------------------------------------
# Transport operators
# ---------------------------------------------------------------------------


def _apply_operator(tensor: torch.Tensor, operator: torch.Tensor, dim: int) -> torch.Tensor:
    if tensor.shape[dim] != operator.shape[0]:
        raise ValueError(
            f"Cannot contract axis {dim} of shape {tuple(tensor.shape)} with a "
            f"{tuple(operator.shape)} transport operator"
        )
    dtype = tensor.dtype if tensor.dtype == torch.float64 else torch.float32
    moved = tensor.movedim(dim, -1)
    result = moved.to(dtype) @ operator.to(device=tensor.device, dtype=dtype)
    return result.movedim(-1, dim).to(tensor.dtype)


@dataclass(frozen=True)
class TransportOperators:
    """The analysis/synthesis pair induced by one transport plan."""

    analysis: torch.Tensor
    synthesis: torch.Tensor

    def __post_init__(self) -> None:
        if self.analysis.shape != self.synthesis.shape:
            raise ValueError(
                "Analysis and synthesis operators must share a shape, got "
                f"{tuple(self.analysis.shape)} and {tuple(self.synthesis.shape)}"
            )

    @property
    def source_size(self) -> int:
        return int(self.analysis.shape[0])

    @property
    def target_size(self) -> int:
        return int(self.analysis.shape[1])

    def project_out(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Map an output axis: a weighted average over source units."""

        return _apply_operator(tensor, self.analysis, dim)

    def project_in(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """Map an input axis: scatter each source unit over its targets."""

        return _apply_operator(tensor, self.synthesis, dim)

    def self_mass(self) -> torch.Tensor:
        """Diagonal of ``synthesis @ analysis.T``, without forming the matrix."""

        return (self.synthesis * self.analysis).sum(dim=1)

    def permutation(self) -> torch.Tensor:
        """Which source unit each target unit draws most of its mass from."""

        return self.analysis.argmax(dim=0)


def transport_operators_from_plan(plan: torch.Tensor) -> TransportOperators:
    plan = plan.double()
    analysis = plan / plan.sum(dim=0, keepdim=True).clamp_min(_EPSILON)
    synthesis = plan / plan.sum(dim=1, keepdim=True).clamp_min(_EPSILON)
    return TransportOperators(analysis, synthesis)


def expanded_operators(operators: TransportOperators, factor: int) -> TransportOperators:
    """Repeat every unit ``factor`` times, keeping the within-unit basis fixed.

    Head plans are expanded by the head dimension: mixing q/k rows across heads
    would break the `q . k` bilinear form, so a head moves whole or not at all.
    """

    identity = torch.eye(int(factor), dtype=operators.analysis.dtype)
    return TransportOperators(
        torch.kron(operators.analysis, identity),
        torch.kron(operators.synthesis, identity),
    )


def identity_operators(size: int) -> TransportOperators:
    identity = torch.eye(size, dtype=torch.float64)
    return TransportOperators(identity, identity)


# ---------------------------------------------------------------------------
# Transport plans
# ---------------------------------------------------------------------------


def barycentric_plan(cost: torch.Tensor, *, max_iterations: int = 1_000_000) -> torch.Tensor:
    """Exact discrete OT between two uniform neuron distributions."""

    ot = _require_pot()
    source_size, target_size = cost.shape
    plan, log = ot.emd(
        np.full(source_size, 1.0 / source_size),
        np.full(target_size, 1.0 / target_size),
        np.ascontiguousarray(cost.double().cpu().numpy()),
        numItermax=int(max_iterations),
        log=True,
    )
    if log["warning"] is not None:
        raise RuntimeError(f"Exact OT solver failed: {log['warning']}")
    return torch.from_numpy(plan).double()


def integral_plan(cost: torch.Tensor) -> torch.Tensor:
    """Transport every target unit's whole mass to one source unit.

    Rounds of rectangular linear-sum assignment give each source unit either
    `floor(n_tgt / n_src)` or `ceil(n_tgt / n_src)` targets, which for the equal
    sizes here means exactly one: a permutation, in one round.
    """

    source_size, target_size = cost.shape
    matrix = cost.double().cpu().numpy()
    remaining = np.arange(target_size)
    owners = np.empty(target_size, dtype=np.int64)
    while remaining.size:
        rows, columns = linear_sum_assignment(matrix[:, remaining])
        owners[remaining[columns]] = rows
        remaining = np.delete(remaining, columns)
    plan = torch.zeros(source_size, target_size, dtype=torch.float64)
    plan[torch.from_numpy(owners), torch.arange(target_size)] = 1.0 / target_size
    return plan


def solve_transport_plan(cost: torch.Tensor, *, column_assignment: str = "integral") -> torch.Tensor:
    if column_assignment not in COLUMN_ASSIGNMENTS:
        choices = ", ".join(sorted(COLUMN_ASSIGNMENTS))
        raise ValueError(
            f"Invalid column assignment {column_assignment!r}; expected one of: {choices}"
        )
    if column_assignment == "integral":
        return integral_plan(cost)
    return barycentric_plan(cost)


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------


class AlignmentStatistics:
    """Streaming cross-moments between source and target unit activations."""

    def __init__(self, source_size: int, target_size: int) -> None:
        self.cross = torch.zeros(source_size, target_size, dtype=torch.float64)
        self.source_sum = torch.zeros(source_size, dtype=torch.float64)
        self.target_sum = torch.zeros(target_size, dtype=torch.float64)
        self.source_square = torch.zeros(source_size, dtype=torch.float64)
        self.target_square = torch.zeros(target_size, dtype=torch.float64)
        self.features = 0

    @torch.no_grad()
    def update(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one ``(units, features)`` pair sharing a feature axis.

        Per-batch moments in float32 on whatever device the activations live on,
        accumulated in float64 on the host: a float64 matmul over 4096-wide
        activations is orders of magnitude slower on a consumer GPU, and the
        ground metric that comes out is a similarity in [0, 2].
        """

        if source.shape[1] != target.shape[1]:
            raise ValueError(
                "Source and target activations must share a feature axis, got "
                f"{source.shape[1]} and {target.shape[1]}"
            )
        source = _rms_normalized(source.float())
        target = _rms_normalized(target.float())
        self.cross += (source @ target.t()).double().cpu()
        self.source_sum += source.sum(dim=1).double().cpu()
        self.target_sum += target.sum(dim=1).double().cpu()
        self.source_square += source.square().sum(dim=1).double().cpu()
        self.target_square += target.square().sum(dim=1).double().cpu()
        self.features += int(source.shape[1])

    def cost(self, metric: str = "correlation") -> torch.Tensor:
        """Ground metric in ``[0, 2]``; smaller means more alike."""

        if metric not in GROUND_METRICS:
            choices = ", ".join(sorted(GROUND_METRICS))
            raise ValueError(f"Invalid ground metric {metric!r}; expected one of: {choices}")
        if self.features == 0:
            raise ValueError("No activations were accumulated")
        cross = self.cross
        source_square = self.source_square
        target_square = self.target_square
        if metric == "correlation":
            cross = cross - torch.outer(self.source_sum, self.target_sum) / self.features
            source_square = source_square - self.source_sum.square() / self.features
            target_square = target_square - self.target_sum.square() / self.features
        scale = torch.outer(
            source_square.clamp_min(_EPSILON).sqrt(),
            target_square.clamp_min(_EPSILON).sqrt(),
        )
        return 1.0 - cross / scale


def _rms_normalized(tensor: torch.Tensor) -> torch.Tensor:
    """Rescale one hook point's activations to unit RMS.

    Both ground metrics are scale-invariant per unit, so this matters only for
    the residual group, whose statistics accumulate all twenty-five block
    boundaries into one cost. Residual norms grow with depth, and without this
    the deepest blocks would decide the alignment on their own.
    """

    scale = tensor.square().mean(dtype=torch.float64).sqrt().to(tensor.dtype)
    return tensor / scale.clamp_min(_EPSILON)


def as_unit_matrix(tensor: torch.Tensor) -> torch.Tensor:
    """`(batch, frames, units)` activations as a `(units, features)` matrix."""

    return tensor.reshape(-1, tensor.shape[-1]).t().contiguous()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def identity_agreement(operators: TransportOperators) -> float:
    """Fraction of units the plan leaves where they were.

    One means the plan is the identity and optimal transport has found nothing a
    permutation could fix. That is the expected answer between two descendants of
    one checkpoint, and it is the number that decides whether the rest of this
    package is about permutations or about the interface.
    """

    if operators.source_size != operators.target_size:
        return float("nan")
    permutation = operators.permutation()
    return float((permutation == torch.arange(operators.target_size)).double().mean())


def summarize(
    group: str,
    cost: torch.Tensor,
    plan: torch.Tensor,
    operators: TransportOperators,
) -> dict[str, Any]:
    transported = float((cost * plan).sum())
    square = min(cost.shape)
    diagonal = float(cost.diagonal()[:square].mean()) if cost.shape[0] == cost.shape[1] else float("nan")
    self_mass = operators.self_mass()
    return {
        "group": group,
        "source_size": operators.source_size,
        "target_size": operators.target_size,
        # cost of the plan, against the cost of doing nothing: if these are the
        # same, the neurons were already numbered the same way
        "transport_cost": transported,
        "identity_cost": diagonal,
        "identity_agreement": identity_agreement(operators),
        "mean_self_mass": float(self_mass.mean()),
        "min_self_mass": float(self_mass.min()),
    }


def solve_group(
    group: str,
    statistics: AlignmentStatistics,
    *,
    ground_metric: str = "correlation",
    column_assignment: str = "integral",
) -> tuple[TransportOperators, dict[str, Any]]:
    cost = statistics.cost(ground_metric)
    plan = solve_transport_plan(cost, column_assignment=column_assignment)
    operators = transport_operators_from_plan(plan)
    return operators, summarize(group, cost, plan, operators)


def concatenated_operators(operators: Sequence[TransportOperators]) -> TransportOperators:
    """Operators for an axis that is the concatenation of several groups."""

    return TransportOperators(
        torch.block_diag(*[operator.analysis for operator in operators]),
        torch.block_diag(*[operator.synthesis for operator in operators]),
    )
