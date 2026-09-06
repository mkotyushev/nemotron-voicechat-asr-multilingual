"""Closed-form RegMean++ merging of ``PT_ML`` and ``FT_EN`` -- Comparison 6.

RegMean (Jin et al., ICLR 2023) replaces the unweighted average of two
fine-tunes with a per-linear-layer least squares against each candidate's own
outputs::

    W_M = [sum_i G_i]^-1 sum_i G_i W_i,
    G_i = alpha * X_i^T X_i + (1 - alpha) * diag(X_i^T X_i)

RegMean++ (Nguyen et al., TMLR 2026, ``papers/2508.03121v3.pdf``) changes only
where ``X_i`` comes from: the *cross-layer* input is the already-merged prefix's
output, and only the *intra-layer* sub-module inputs come from running candidate
``i``'s own layer on it.  Everything else -- including the fact that there is no
gradient anywhere and no ground truth anywhere -- is unchanged.

Three properties of this encoder decide most of what follows, and
``REGMEAN_INTERFACE_DESIGN.md`` argues each one:

  * **The Gram matrices must come from different data.**  Eq. 2 is a
    ``G``-weighted average of the candidate weights, so equal Grams collapse it
    exactly onto the unweighted mean.  ``G_F`` is therefore collected on English
    assistant audio and ``G_M`` on multilingual audio, and :func:`regmean_solve`
    reports the collapse when it happens anyway -- which it does for
    ``relative_k_proj``, whose input is the position encoding rather than the
    data.
  * **1x1 convolutions are dense linears.**  ``conv.pointwise_conv{1,2}`` are
    ``Conv1d(k=1)`` over channels and belong on the RegMean path; leaving them on
    the averaging path is the easy mistake.  ``depthwise_conv``, the subsampling
    convolutions, every LayerNorm, ``bias_u`` and ``bias_v`` are averaged.
  * **Every tensor is routed exactly once.**  :func:`route_tensors` is the
    manifest Comparison 6 requires, and it fails rather than silently leaving a
    tensor at its ``PT_ML`` value.

Nothing here controls error at the *encoder output*, which is the only quantity
the frozen language model reads: each depth is solved greedily against its own
inputs.  That is why :func:`output_agreement` exists as a separate, held-out
measurement and why the alpha grid is selected on it rather than on the
per-layer residuals.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .encoder import (
    Block,
    Encoder,
    Hyper,
    Subsampling,
    _attention_mask,
    _position_embedding,
    _relative_shift_index,
)
from .experiments import CANONICAL_PREFIX, ExperimentValidationError, stable_json_sha256

COMPARISON = 6
ARTIFACT_KIND = "regmean-merge"

#: The design record's grid.  ``alpha = 1.0`` -- no shrinkage at all -- zeroes
#: out accuracy in the paper's Table 9 and is excluded deliberately.
ALPHAS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.95)

#: Rows per input dimension the Gram set must reach.  The paper's literal "256
#: samples" is 256 images x ~50 patches against a largest ``d_in`` of 3072, i.e.
#: this ratio; copying the sample count instead of the ratio would solve
#: ``subsampling.linear`` (``d_in`` = 4352) from a Gram of 2.2 rows per
#: dimension and let the shrinkage silently do the work.
GRAM_ROWS_PER_INPUT_DIMENSION = 4

#: Candidate symbols, in the order Eq. 2 sums over them.
CANDIDATES = ("M", "F")

MODULE_KINDS = ("subsampling", "attention", "ffn", "conv")
LAYERNORM_SOURCES = ("average", "F")
ROUTES = ("regmean", "average")

#: Ridge on Eq. 2's solve, relative to the regularized Gram's leading
#: eigenvalue.  The Gram is accumulated from F32 inner products of F32
#: activations, so directions below roughly 1e-6 of the top eigenvalue carry no
#: signal; independently, a 17k-row sample cannot resolve a 4096-dimensional
#: second moment that far down.  Both arguments point at the same floor.
RIDGE_RELATIVE = 1e-6


class MergeValidationError(ExperimentValidationError):
    """A merge invariant was violated."""


@dataclass(frozen=True)
class LinearSite:
    """One dense linear Eq. 2 solves for, and where to find its input.

    ``depth`` is 0 for the subsampling projection and ``block + 1`` for anything
    inside encoder block ``block``, which is the order RegMean++ visits them in.
    ``module`` is the dotted path *within that depth's root module* -- the
    ``Subsampling`` stem or one ``Block`` -- because that is the granularity the
    merge runs the graph at.  ``conv1d`` marks the two pointwise convolutions,
    whose captured input is ``(batch, channels, frames)`` rather than
    frames-last.
    """

    tensor: str
    module: str
    depth: int
    kind: str
    in_features: int
    out_features: int
    conv1d: bool = False

    @property
    def block(self) -> int | None:
        return None if self.depth == 0 else self.depth - 1


def hyper_from_config(config: Mapping[str, Any], subsampling_channels: int) -> Hyper:
    """The graph's hyper-parameters, read the way :func:`asr_align.encoder.build` reads them."""

    return Hyper(
        n_layer=int(config["num_hidden_layers"]),
        n_embd=int(config["hidden_size"]),
        n_head=int(config["num_attention_heads"]),
        n_ff=int(config["intermediate_size"]),
        n_mel=int(config["num_mel_bins"]),
        conv_kernel=int(config["conv_kernel_size"]),
        attention_left_context=int(config["sliding_window"]) - 1,
        subsampling_channels=subsampling_channels,
    )


def subsampling_input_dim(hyper: Hyper) -> int:
    """``subsampling.linear``'s input width: channels times surviving mel bins."""

    n_freq = hyper.n_mel
    for _ in range(3):
        n_freq = n_freq // 2 + 1
    return hyper.subsampling_channels * n_freq


def linear_sites(hyper: Hyper) -> tuple[LinearSite, ...]:
    """Every dense linear in the encoder, in depth order."""

    sites = [
        LinearSite(
            tensor=f"{CANONICAL_PREFIX}subsampling.linear.weight",
            module="linear",
            depth=0,
            kind="subsampling",
            in_features=subsampling_input_dim(hyper),
            out_features=hyper.n_embd,
        )
    ]
    for index in range(hyper.n_layer):
        prefix = f"{CANONICAL_PREFIX}layers.{index}."
        depth = index + 1
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj"):
            sites.append(
                LinearSite(
                    tensor=f"{prefix}self_attn.{name}.weight",
                    module=f"self_attn.{name}",
                    depth=depth,
                    kind="attention",
                    in_features=hyper.n_embd,
                    out_features=hyper.n_embd,
                )
            )
        for which in (1, 2):
            sites.append(
                LinearSite(
                    tensor=f"{prefix}feed_forward{which}.linear1.weight",
                    module=f"feed_forward{which}.linear1",
                    depth=depth,
                    kind="ffn",
                    in_features=hyper.n_embd,
                    out_features=hyper.n_ff,
                )
            )
            sites.append(
                LinearSite(
                    tensor=f"{prefix}feed_forward{which}.linear2.weight",
                    module=f"feed_forward{which}.linear2",
                    depth=depth,
                    kind="ffn",
                    in_features=hyper.n_ff,
                    out_features=hyper.n_embd,
                )
            )
        sites.append(
            LinearSite(
                tensor=f"{prefix}conv.pointwise_conv1.weight",
                module="conv.pointwise_conv1",
                depth=depth,
                kind="conv",
                in_features=hyper.n_embd,
                out_features=2 * hyper.n_embd,
                conv1d=True,
            )
        )
        sites.append(
            LinearSite(
                tensor=f"{prefix}conv.pointwise_conv2.weight",
                module="conv.pointwise_conv2",
                depth=depth,
                kind="conv",
                in_features=hyper.n_embd,
                out_features=hyper.n_embd,
                conv1d=True,
            )
        )
    return tuple(sites)


@dataclass(frozen=True)
class MergePlan:
    """The declared choices behind one merged candidate.

    ``depths`` and ``kinds`` are the paper's contribution (2): merging only the
    middle and deep transformer layers preserves >98% of the all-layer result,
    and MLP linears outperform attention linears.  They are recorded per
    candidate rather than defaulted silently.  A RegMean-path tensor outside the
    declared subset falls back to averaging, which is what the paper's own
    ablation does with it.
    """

    alpha: float
    cross_layer: bool = True
    depths: tuple[int, ...] | None = None
    kinds: tuple[str, ...] = MODULE_KINDS
    layernorm_source: str = "average"

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise MergeValidationError(
                f"alpha={self.alpha!r} must lie strictly between 0 and 1; "
                "alpha=1 removes the shrinkage the method depends on"
            )
        unknown = sorted(set(self.kinds) - set(MODULE_KINDS))
        if unknown or not self.kinds:
            raise MergeValidationError(f"unknown module kinds {unknown}")
        if self.layernorm_source not in LAYERNORM_SOURCES:
            raise MergeValidationError(
                f"layernorm_source must be one of {LAYERNORM_SOURCES}"
            )

    @property
    def method(self) -> str:
        return "regmean++" if self.cross_layer else "regmean"

    def includes(self, site: LinearSite) -> bool:
        if site.kind not in self.kinds:
            return False
        return self.depths is None or site.depth in self.depths

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "cross_layer_input": "merged prefix" if self.cross_layer else "candidate's own",
            "merged_depths": "all" if self.depths is None else list(self.depths),
            "merged_module_kinds": list(self.kinds),
            "layernorm_source": self.layernorm_source,
            "non_linear_tensors": "simple average",
        }


def route_tensors(
    keys: Iterable[str], hyper: Hyper, plan: MergePlan
) -> dict[str, Any]:
    """Classify every canonical encoder tensor, exactly once.

    Comparison 6 requires the assertion, not just the classification: a tensor
    that fell through both paths would keep whichever candidate's value happened
    to be copied first, and nothing downstream would notice.
    """

    keys = sorted(keys)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise MergeValidationError(f"tensor appears twice in the state dict: {duplicates}")
    sites = {site.tensor: site for site in linear_sites(hyper)}
    missing = sorted(set(sites) - set(keys))
    if missing:
        raise MergeValidationError(f"the encoder state lacks dense linears {missing}")
    routes: dict[str, dict[str, Any]] = {}
    for key in keys:
        if not key.startswith(CANONICAL_PREFIX):
            raise MergeValidationError(f"{key} is outside the canonical encoder namespace")
        site = sites.get(key)
        if site is None:
            reason = "not a dense linear"
            if key.endswith(("norm.weight", "norm.bias")) or ".norm_" in key:
                reason = "LayerNorm"
            elif "depthwise_conv" in key or "subsampling.conv" in key:
                reason = "grouped or spatial convolution, no dense form"
            elif key.endswith(("bias_u", "bias_v")):
                reason = "attention position bias"
            elif key.endswith(".bias"):
                reason = "bias; Eq. 2 solves the weight matrix only"
            routes[key] = {"route": "average", "reason": reason}
        elif plan.includes(site):
            routes[key] = {
                "route": "regmean",
                "depth": site.depth,
                "kind": site.kind,
                "in_features": site.in_features,
                "out_features": site.out_features,
            }
        else:
            routes[key] = {
                "route": "average",
                "reason": "dense linear outside the declared depth/module subset",
                "depth": site.depth,
                "kind": site.kind,
            }
    counts = {route: sum(1 for value in routes.values() if value["route"] == route) for route in ROUTES}
    if counts["regmean"] + counts["average"] != len(keys):
        raise MergeValidationError("routing does not cover every tensor exactly once")
    return {
        "plan": plan.as_dict(),
        "count": len(keys),
        "counts": counts,
        "tensors": routes,
        "tensors_sha256": stable_json_sha256(routes),
    }


@dataclass
class GramAccumulator:
    """``X^T X`` for one linear site, normalized by the rows that produced it.

    Normalizing is not cosmetic.  Speech clips are not fixed-size samples, so
    without it a 30 s clip contributes four times a 7 s clip and the clip-length
    distribution becomes an unintended merge coefficient.
    """

    in_features: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    rows: int = 0
    _moment: torch.Tensor | None = None

    def update(self, value: torch.Tensor) -> None:
        # Each batch's product is F32 -- a few hundred rows, so the within-batch
        # error is around 1e-6 relative -- and the running sum is F64, which is
        # where accumulating tens of thousands of rows would otherwise lose
        # digits.  Two candidates fed identical inputs still accumulate
        # identically, which is what makes the collapse-to-the-mean check exact.
        flat = value.reshape(-1, value.shape[-1]).float()
        if flat.shape[-1] != self.in_features:
            raise MergeValidationError(
                f"Gram input width {flat.shape[-1]} does not match the layer's {self.in_features}"
            )
        if not bool(torch.isfinite(flat).all()):
            raise MergeValidationError("Gram input contains NaN or infinity")
        if self._moment is None:
            self._moment = torch.zeros(
                self.in_features, self.in_features, dtype=torch.float64, device=flat.device
            )
        self._moment += (flat.T @ flat).double()
        self.rows += int(flat.shape[0])

    def gram(self) -> torch.Tensor:
        if self._moment is None or self.rows == 0:
            raise MergeValidationError("Gram accumulator saw no rows")
        value = self._moment / float(self.rows)
        return 0.5 * (value + value.T)


def shrink(gram: torch.Tensor, alpha: float) -> torch.Tensor:
    """``alpha * G + (1 - alpha) * diag(G)``, Eq. 2's regularized Gram."""

    return alpha * gram + (1.0 - alpha) * torch.diag(torch.diagonal(gram))


def _relative_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    scale = max(float(left.norm()), float(right.norm()), 1e-30)
    return float((left - right).norm()) / scale


def _gram_relative_residual(
    gram: torch.Tensor, merged: torch.Tensor, own: torch.Tensor
) -> float:
    """``||X (W_M - W_i)||_F / ||X W_i||_F`` from the normalized Gram alone."""

    delta = (merged.to(dtype=torch.float64) - own.to(dtype=torch.float64)).T
    reference = own.to(dtype=torch.float64).T
    numerator = float((delta * (gram @ delta)).sum())
    denominator = float((reference * (gram @ reference)).sum())
    return (max(numerator, 0.0) / max(denominator, 1e-30)) ** 0.5


def _leading_eigenvalue(matrix: torch.Tensor, iterations: int = 32) -> float:
    """Power iteration for the largest eigenvalue of an SPD matrix.

    The ridge below is relative to this, and an eigendecomposition just to find
    it would cost more than the solve.
    """

    vector = torch.ones(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    vector /= vector.norm()
    value = 0.0
    for _ in range(iterations):
        product = matrix @ vector
        norm = float(product.norm())
        if norm <= 0.0:
            return 0.0
        vector = product / norm
        value = norm
    return value


def regmean_solve(
    grams: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    alpha: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Eq. 2 for one linear, returning the ``(out, in)`` weight and diagnostics.

    ``weights`` are ``nn.Linear``-shaped ``(out, in)``; Eq. 2's ``W`` is their
    transpose, and ``G`` is over the input dimension.

    **The solve is for the offset from the candidates' mean, and it is ridged.**
    That is not decoration.  Writing ``W_M = W_bar + D`` and subtracting, Eq. 2
    becomes

        [sum_i G_i] D = sum_i G_i (W_i - W_bar) = (G_F - G_M)(W_F - W_M) / 2,

    so ``D`` is decided entirely by the *difference* between the two Gram
    matrices -- which is the quantity RegMean exists to exploit, and also the
    quantity that is pure round-off wherever the encoder's activations do not
    reach.  Speech activations here are strongly low-rank (``subsampling.linear``
    reads 4352 post-ReLU features; the FFNs read 4096), the shrinkage term
    ``(1 - alpha) diag(G)`` vanishes in exactly the directions that are dead in
    *both* candidates, and solving unridged produced encoder weights 8e14 times
    ``PT_ML``'s norm: a finite, plausible-looking answer built entirely out of
    noise.  Ridging the offset leaves ``W_M`` at the candidates' mean in those
    directions and unchanged where the data actually determines it, which is the
    minimum-deviation choice among the objective's minimizers rather than a
    different objective.  ``ridge_relative`` and the resulting effective rank are
    recorded per layer.
    """

    if set(grams) != set(weights) or not grams:
        raise MergeValidationError("Eq. 2 needs one Gram per candidate weight")
    order = sorted(grams)
    shrunk = {name: shrink(grams[name], alpha) for name in order}
    total = sum(shrunk.values())
    mean = torch.stack([weights[name].to(dtype=torch.float64) for name in order]).mean(dim=0)
    offset = sum(
        shrunk[name] @ (weights[name].to(dtype=torch.float64) - mean).T for name in order
    )

    largest = _leading_eigenvalue(total)
    if largest <= 0.0:
        raise MergeValidationError("Eq. 2's regularized Gram is identically zero")
    eye = torch.eye(total.shape[0], dtype=total.dtype, device=total.device)
    ridge_relative = RIDGE_RELATIVE
    for _ in range(6):
        ridge = ridge_relative * largest
        factor, info = torch.linalg.cholesky_ex(total + ridge * eye)
        if int(info) == 0:
            solution = torch.cholesky_solve(offset, factor)
            if bool(torch.isfinite(solution).all()):
                break
        ridge_relative *= 100.0
    else:
        raise MergeValidationError("Eq. 2 did not produce a finite solution")

    spectrum = torch.linalg.eigvalsh(total)
    smallest = float(spectrum[0])
    condition = largest / smallest if smallest > 0 else None
    # How many directions the data actually determines.  The floor is the ridge,
    # so this is exactly the subspace the solve was allowed to move in.
    effective_rank = int((spectrum > ridge_relative * largest).sum())

    merged = (mean.T + solution).T.contiguous()
    # The Grams coincide whenever the layer's input does not depend on the data
    # -- `relative_k_proj` reads the position encoding -- and Eq. 2 then reduces
    # exactly to the unweighted mean.  That is worth recording, not hiding.
    pairs = [
        _relative_difference(shrunk[order[i]], shrunk[order[j]])
        for i in range(len(order))
        for j in range(i + 1, len(order))
    ]
    diagnostics = {
        "alpha": alpha,
        "ridge_relative": ridge_relative,
        "ridge": ridge_relative * largest,
        "ridge_target": "the candidates' mean",
        "condition_number": condition,
        "eigenvalue_range": [smallest, largest],
        "effective_rank": effective_rank,
        "input_dimension": int(total.shape[0]),
        "gram_relative_difference": max(pairs) if pairs else 0.0,
        "distance_from_mean": _relative_difference(merged, mean),
        "distance_from_candidate": {
            name: _relative_difference(merged, weights[name].to(dtype=torch.float64))
            for name in order
        },
        # The quantity Eq. 2 actually minimizes, read straight off the
        # unshrunk Gram: ||X (W_M - W_i)||_F / ||X W_i||_F needs no second pass
        # over the inputs because both are traces against X^T X.
        "output_residual_relative": {
            name: _gram_relative_residual(grams[name], merged, weights[name])
            for name in order
        },
    }
    diagnostics["reduces_to_mean"] = bool(
        diagnostics["gram_relative_difference"] < 1e-9
        and diagnostics["distance_from_mean"] < 1e-6
    )
    return merged.to(dtype=torch.float32), diagnostics


def simple_average(states: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Arm ``E4``'s encoder, and the fallback for every non-linear tensor."""

    names = sorted(states)
    if len(names) < 2:
        raise MergeValidationError("averaging needs at least two candidates")
    keys = set(states[names[0]])
    for name in names[1:]:
        if set(states[name]) != keys:
            raise MergeValidationError(f"{name} tensor keys differ from {names[0]}")
    out: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        shapes = {name: tuple(states[name][key].shape) for name in names}
        if len(set(shapes.values())) != 1:
            raise MergeValidationError(
                f"shape mismatch for {key}: {shapes}; broadcasting is forbidden"
            )
        stacked = torch.stack([states[name][key].to(dtype=torch.float32) for name in names])
        value = stacked.mean(dim=0)
        if not bool(torch.isfinite(value).all()):
            raise MergeValidationError(f"average of {key} is not finite")
        out[key] = value.contiguous()
    return out



def _strip(state: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)
    }


def _subsampling(hyper: Hyper, state: Mapping[str, torch.Tensor]) -> Subsampling:
    module = Subsampling(hyper)
    module.load_state_dict(_strip(state, f"{CANONICAL_PREFIX}subsampling."), strict=True)
    return module.eval()


def _block(hyper: Hyper, state: Mapping[str, torch.Tensor], index: int) -> Block:
    module = Block(hyper)
    module.load_state_dict(_strip(state, f"{CANONICAL_PREFIX}layers.{index}."), strict=True)
    return module.eval()


def depth_of(key: str) -> int:
    """0 for the subsampling stem, ``block + 1`` inside a block: RegMean++'s visit order."""

    rest = key[len(CANONICAL_PREFIX):]
    if rest.startswith("subsampling."):
        return 0
    if rest.startswith("layers."):
        return int(rest.split(".")[1]) + 1
    raise MergeValidationError(f"cannot place {key} at a depth")


def is_layer_norm(key: str) -> bool:
    tail = key.rsplit(".", 1)[0]
    return tail.rsplit(".", 1)[-1] in {"norm", "norm_self_att", "norm_out", "norm_conv"} or (
        ".norm_feed_forward" in key
    )


def _attention_arguments(
    time: int, hyper: Hyper, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _position_embedding(time, hyper.n_embd, device, dtype),
        _attention_mask(time, hyper.attention_left_context, device, dtype),
        _relative_shift_index(time, device),
    )


def _capture(module: torch.nn.Module, sites: Sequence[LinearSite], store: dict) -> list:
    """Forward pre-hooks recording each site's input, transposing the 1x1 convolutions."""

    handles = []
    for site in sites:
        target = module
        for part in site.module.split("."):
            target = getattr(target, part)

        def hook(_module, args, site=site):
            value = args[0]
            store[site.tensor] = value.transpose(1, 2) if site.conv1d else value

        handles.append(target.register_forward_pre_hook(hook))
    return handles


def _as_linear(site: LinearSite, tensor: torch.Tensor) -> torch.Tensor:
    """``(out, in)``, collapsing the trailing width-1 axis of a 1x1 convolution."""

    return tensor.reshape(site.out_features, site.in_features) if site.conv1d else tensor


@torch.inference_mode()
def merge_encoders(
    states: Mapping[str, Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
    mel_batches: Mapping[str, Sequence[torch.Tensor]],
    plan: MergePlan,
    *,
    device: torch.device | str = "cpu",
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """RegMean++ Algorithm 1 over the whole encoder, depth by depth.

    Each candidate carries its own Gram audio -- that difference is the only
    thing Eq. 2 has to work with -- so ``mel_batches`` is keyed by candidate and
    the two sets are never mixed.  At every depth the cross-layer input comes
    from the already-merged prefix and the intra-layer sub-module inputs come
    from running that candidate's own layer on it; with ``cross_layer=False``
    both come from the candidate's own forward pass, which is plain RegMean.

    Everything runs under ``config`` -- ``PT_ML``'s complete runtime
    configuration, 56-frame left context included -- so ``FT_EN``'s Gram
    contribution is collected at a context it never saw.  Invariant 5 requires
    that, and the design record records the consequence rather than working
    around it.

    Grams accumulate on ``device`` in F64; the solves run on the CPU, where F64
    is not a sixty-fourth of the throughput.
    """

    device = torch.device(device)
    names = sorted(states)
    if names != sorted(CANDIDATES):
        raise MergeValidationError(f"the merge takes exactly {sorted(CANDIDATES)}, got {names}")
    if sorted(mel_batches) != names:
        raise MergeValidationError("every candidate needs its own Gram audio")
    keys = set(states[names[0]])
    for name in names[1:]:
        if set(states[name]) != keys:
            raise MergeValidationError(f"{name} tensor keys differ from {names[0]}")

    channels = int(states[names[0]][f"{CANONICAL_PREFIX}subsampling.conv_in.weight"].shape[0])
    hyper = hyper_from_config(config, channels)
    sites = linear_sites(hyper)
    routing = route_tensors(keys, hyper, plan)
    averaged = simple_average(states)

    def non_linear(key: str) -> torch.Tensor:
        # Under ++ the averaged non-linear tensors feed every downstream solve,
        # which is why seeding the LayerNorms from F is an ablation worth having
        # and why it has to happen here rather than as a post-pass.
        if plan.layernorm_source == "F" and is_layer_norm(key):
            return states["F"][key].to(dtype=torch.float32).clone()
        return averaged[key].clone()

    merged: dict[str, torch.Tensor] = {}
    site_reports: dict[str, Any] = {}
    gram_rows: dict[str, int] = {}

    def solve_depth(
        depth: int, accumulators: Mapping[str, Mapping[str, GramAccumulator]]
    ) -> None:
        for site in sites:
            if site.depth != depth:
                continue
            if not plan.includes(site):
                merged[site.tensor] = averaged[site.tensor].clone()
                continue
            grams = {
                name: accumulators[name][site.tensor].gram().cpu() for name in names
            }
            rows = {name: accumulators[name][site.tensor].rows for name in names}
            weight, diagnostics = regmean_solve(
                grams,
                {
                    name: _as_linear(site, states[name][site.tensor]).cpu().double()
                    for name in names
                },
                plan.alpha,
            )
            merged[site.tensor] = weight.reshape(
                states[names[0]][site.tensor].shape
            ).contiguous()
            diagnostics.update(
                rows=rows,
                depth=site.depth,
                kind=site.kind,
                in_features=site.in_features,
                rows_per_input_dimension={
                    name: rows[name] / site.in_features for name in names
                },
            )
            site_reports[site.tensor] = diagnostics
        for key in sorted(keys):
            if key not in merged and depth_of(key) == depth:
                merged[key] = non_linear(key)

    def collect(
        module: torch.nn.Module,
        depth_sites: Sequence[LinearSite],
        batches: Sequence[torch.Tensor],
        run: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[dict[str, GramAccumulator], list[torch.Tensor]]:
        accumulators = {
            site.tensor: GramAccumulator(site.in_features) for site in depth_sites
        }
        store: dict[str, torch.Tensor] = {}
        handles = _capture(module, depth_sites, store)
        outputs: list[torch.Tensor] = []
        try:
            for value in batches:
                store.clear()
                outputs.append(run(value))
                for site in depth_sites:
                    accumulators[site.tensor].update(store[site.tensor])
        finally:
            for handle in handles:
                handle.remove()
        return accumulators, outputs

    # ---- depth 0: the subsampling projection -----------------------------
    depth_sites = [site for site in sites if site.depth == 0]
    accumulators: dict[str, dict[str, GramAccumulator]] = {}
    own_outputs: dict[str, list[torch.Tensor]] = {}
    for name in names:
        module = _subsampling(hyper, states[name]).to(device)
        accumulators[name], own_outputs[name] = collect(
            module,
            depth_sites,
            [mel.to(device) for mel in mel_batches[name]],
            module,
        )
        gram_rows[name] = accumulators[name][depth_sites[0].tensor].rows
        del module
        if progress is not None:
            progress(f"depth 0 Gram: {name}, {gram_rows[name]} rows")
    solve_depth(0, accumulators)
    del accumulators

    streams: dict[str, list[torch.Tensor]] = {}
    if plan.cross_layer:
        merged_subsampling = _subsampling(hyper, merged).to(device)
        for name in names:
            streams[name] = [
                merged_subsampling(mel.to(device)) for mel in mel_batches[name]
            ]
        del merged_subsampling
    else:
        streams = own_outputs
    del own_outputs

    # ---- depths 1..L: the encoder blocks ---------------------------------
    for index in range(hyper.n_layer):
        depth = index + 1
        depth_sites = [site for site in sites if site.depth == depth]
        accumulators = {}
        own_outputs = {}
        for name in names:
            block = _block(hyper, states[name], index).to(device)

            def run(value: torch.Tensor, block: Block = block) -> torch.Tensor:
                return block(
                    value, *_attention_arguments(value.shape[1], hyper, device, value.dtype)
                )

            accumulators[name], own_outputs[name] = collect(
                block, depth_sites, streams[name], run
            )
            del block
        solve_depth(depth, accumulators)
        del accumulators
        if plan.cross_layer:
            merged_block = _block(hyper, merged, index).to(device)
            for name in names:
                streams[name] = [
                    merged_block(
                        value,
                        *_attention_arguments(value.shape[1], hyper, device, value.dtype),
                    )
                    for value in streams[name]
                ]
            del merged_block, own_outputs
        else:
            streams = own_outputs
        if progress is not None:
            progress(f"depth {depth}/{hyper.n_layer} merged")
    del streams

    if set(merged) != keys:
        missing = sorted(keys - set(merged))
        extra = sorted(set(merged) - keys)
        raise MergeValidationError(
            f"merge left tensors unrouted: missing={missing[:8]}, extra={extra[:8]}"
        )
    for key, value in merged.items():
        if value.shape != states[names[0]][key].shape:
            raise MergeValidationError(f"merged {key} changed shape")
        if not bool(torch.isfinite(value).all()):
            raise MergeValidationError(f"merged {key} is not finite")
        merged[key] = value.detach().to(dtype=torch.float32).cpu().contiguous()

    report = {
        "routing": routing,
        "gram": {
            "rows_per_candidate": dict(gram_rows),
            "normalized_by_frame_count": True,
            "frames_equalized_across_candidates": len(set(gram_rows.values())) == 1,
            "rows_per_input_dimension_target": GRAM_ROWS_PER_INPUT_DIMENSION,
            "widest_linear_input": max(site.in_features for site in sites),
            "rows_per_widest_input_dimension": {
                name: rows / max(site.in_features for site in sites)
                for name, rows in gram_rows.items()
            },
        },
        "runtime_configuration": {
            "source": "M/PT_ML",
            "attention_left_context": hyper.attention_left_context,
            "ft_en_gram_collected_at_unseen_context": True,
        },
        "layers": site_reports,
        "degenerate_solves": sorted(
            key for key, value in site_reports.items() if value["reduces_to_mean"]
        ),
    }
    return merged, report


@torch.inference_mode()
def output_agreement(
    merged: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    mel_batches: Sequence[torch.Tensor],
    *,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Frame-level agreement at the *encoder output*, on held-out audio.

    RegMean++ solves each depth greedily and never looks downstream, so the
    per-layer residuals in :func:`merge_encoders`'s report say nothing about the
    only quantity the frozen language model reads.  This does, which is why the
    alpha grid is selected on it.
    """

    device = torch.device(device)
    channels = int(merged[f"{CANONICAL_PREFIX}subsampling.conv_in.weight"].shape[0])
    hyper = hyper_from_config(config, channels)
    models = []
    for state in (merged, reference):
        model = Encoder(hyper)
        model.load_state_dict(_strip(state, CANONICAL_PREFIX), strict=True)
        models.append(model.eval().to(device))

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for mel in mel_batches:
        mel = mel.to(device)
        left = models[0](mel).reshape(-1, hyper.n_embd).double().cpu()
        right = models[1](mel).reshape(-1, hyper.n_embd).double().cpu()
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise MergeValidationError("held-out encoder output is not finite")
        pairs.append((left, right))
    frames = sum(int(right.shape[0]) for _, right in pairs)
    if frames == 0:
        raise MergeValidationError("held-out agreement saw no frames")
    mean = sum(right.sum(dim=0) for _, right in pairs) / frames

    error = sum(float(torch.square(left - right).sum()) for left, right in pairs)
    total = sum(float(torch.square(right - mean).sum()) for _, right in pairs)
    reference_energy = sum(float(torch.square(right).sum()) for _, right in pairs)
    cosine = sum(
        float(torch.nn.functional.cosine_similarity(left, right, dim=-1).sum())
        for left, right in pairs
    )
    return {
        "r2": 1.0 - error / max(total, 1e-30),
        "cosine_mean": cosine / frames,
        "relative_error": (error / max(reference_energy, 1e-30)) ** 0.5,
        "frames": frames,
    }


def selection_score(agreements: Mapping[str, Mapping[str, float]]) -> float:
    """The declared alpha criterion: mean held-out output R2 over both candidates.

    Symmetric on purpose.  Weighting it toward ``FT_EN`` would select for
    interface compatibility and weighting it toward ``PT_ML`` for multilingual
    retention, and choosing between those before measuring either is the
    question this comparison exists to inform.
    """

    if sorted(agreements) != sorted(CANDIDATES):
        raise MergeValidationError(f"selection needs one agreement per candidate {CANDIDATES}")
    return sum(float(value["r2"]) for value in agreements.values()) / len(agreements)


def delta_table(
    candidate: Mapping[str, Any],
    pt_ml: Mapping[str, Any],
    task_arithmetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Comparison 6 against Comparison 1, and against Comparison 2's endpoint.

    Both differences are reported relative to ``PT_ML`` rather than to each
    other, because that is the only direction with a paired interval: each
    record's intervals were computed against the same frozen Comparison 1
    arrays, and a candidate-minus-candidate difference of two marginal numbers
    would not be one.
    """

    from . import evaluation as evaluation_module

    evaluation_module.validate_result(candidate)
    evaluation_module.validate_result(pt_ml)
    if candidate.get("comparison") != COMPARISON or pt_ml.get("comparison") != 1:
        raise MergeValidationError("the delta table pairs Comparison 6 against Comparison 1")
    others = {"comparison_2_lambda_1": task_arithmetic} if task_arithmetic is not None else {}
    for name, other in others.items():
        evaluation_module.validate_result(other)
        if other.get("comparison") != 2 or float(other.get("lambda", -1)) != 1.0:
            raise MergeValidationError(f"{name} is not Comparison 2's lambda=1 record")
    for name, other in (("comparison_1", pt_ml), *others.items()):
        if other.get("precision") != candidate.get("precision"):
            raise MergeValidationError(f"{name} is a different precision stage")
        if other.get("manifests") != candidate.get("manifests"):
            raise MergeValidationError(f"{name} used different frozen manifests")

    def row(values: Mapping[str, Any], reference: Mapping[str, Any], metric: str) -> dict[str, Any]:
        return {
            "value": float(values[metric]),
            "difference_vs_pt_ml": float(values[metric]) - float(reference[metric]),
            "paired_interval": values["confidence_intervals"]["difference_vs_pt_ml"].get(metric),
        }

    english_reference = pt_ml["evaluations"]["english_voicechat_space"]
    english = {
        metric: {
            "pt_ml": float(english_reference[metric]),
            "merge": row(candidate["evaluations"]["english_voicechat_space"], english_reference, metric),
            **{
                name: row(other["evaluations"]["english_voicechat_space"], english_reference, metric)
                for name, other in others.items()
            },
        }
        for metric in ("r2", "cosine_mean")
    }

    retrieval: dict[str, Any] = {}
    for task in evaluation_module.RETRIEVAL_TASKS:
        groups = candidate["evaluations"][task]["groups"]
        reference_groups = pt_ml["evaluations"][task]["groups"]
        if set(groups) != set(reference_groups):
            raise MergeValidationError(f"{task} groups differ between comparisons 1 and 6")
        rows: dict[str, Any] = {}
        for group, metrics in groups.items():
            reference = reference_groups[group]
            rows[group] = {
                metric: {
                    "pt_ml": float(reference[metric]),
                    "merge": row(metrics, reference, metric),
                    **{
                        name: row(other["evaluations"][task]["groups"][group], reference, metric)
                        for name, other in others.items()
                    },
                }
                for metric in evaluation_module.RETRIEVAL_METRICS
            }
            rows[group]["n"] = int(metrics["n"])
        retrieval[task] = rows

    return {
        "schema_version": "1.0",
        "comparison": COMPARISON,
        "measures": (
            "closed-form merging through the untouched interface, against the "
            "PT_ML baseline and against direct task arithmetic at lambda=1"
        ),
        "direction": "each candidate minus Comparison 1",
        "precision": candidate["precision"],
        "manifests": dict(candidate["manifests"]),
        "english_voicechat_space": english,
        "retrieval": retrieval,
    }
