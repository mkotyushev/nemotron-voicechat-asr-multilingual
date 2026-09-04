"""Shared invariants for every encoder-transfer experiment.

There are three checkpoints and one direction throughout the experiment suite::

    E = PT_EN      M = PT_ML      F = FT_EN
    delta_F = F - E
    C(lambda) = M + lambda * delta_F

This module is intentionally strict.  Model arithmetic is only defined over the
canonical ``encoder.*`` state dict returned by :mod:`asr_align.weights`; exact
keys and shapes are required, operands are made contiguous F32 tensors, and
non-finite inputs or results stop the run.  In particular, PyTorch broadcasting
is never allowed to decide what a checkpoint operation means.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

CANONICAL_PREFIX = "encoder."
ROLE_DEFINITIONS = {
    "E": "PT_EN",
    "M": "PT_ML",
    "F": "FT_EN",
}
EXPECTED_REPO_IDS = {
    "E": "nvidia/nemotron-speech-streaming-en-0.6b",
    "M": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "F": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
}
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIMARY_LAMBDA = 1.0
ARITHMETIC_DTYPE = torch.float32
ARITHMETIC_PRECISION = "float32"


class ExperimentValidationError(ValueError):
    """A shared experimental invariant was violated."""


@dataclass(frozen=True)
class CheckpointIdentity:
    """Pinned identity supplied by the experiment owner, not guessed from a path."""

    role: str
    repo_id: str
    revision: str
    path: Path
    kind: str

    @classmethod
    def from_dict(cls, role: str, value: Mapping[str, Any]) -> "CheckpointIdentity":
        expected = ROLE_DEFINITIONS.get(role)
        if expected is None:
            raise ExperimentValidationError(f"unknown checkpoint role {role!r}")
        declared = value.get("name", expected)
        if declared != expected:
            raise ExperimentValidationError(
                f"role {role} must be named {expected}, not {declared!r}"
            )
        repo_id = str(value.get("repo_id", "")).strip()
        revision = str(value.get("revision", "")).strip()
        path = Path(str(value.get("path", ""))).expanduser()
        kind = str(value.get("kind", "asr"))
        if not repo_id or not revision or not str(value.get("path", "")).strip():
            raise ExperimentValidationError(
                f"{role}/{expected} needs non-empty repo_id, revision, and path"
            )
        if repo_id != EXPECTED_REPO_IDS[role]:
            raise ExperimentValidationError(
                f"{role}/{expected} must identify {EXPECTED_REPO_IDS[role]}, not {repo_id}"
            )
        if revision.lower() in {"main", "master", "latest", "head"} or any(
            token in revision.lower() for token in ("replace", "todo", "unknown")
        ):
            raise ExperimentValidationError(
                f"{role}/{expected} revision {revision!r} is movable; pin a commit or immutable tag"
            )
        allowed_kinds = {"asr", "voicechat_container"}
        if kind not in allowed_kinds:
            raise ExperimentValidationError(
                f"{role}/{expected} kind {kind!r} is not one of {sorted(allowed_kinds)}"
            )
        return cls(role, repo_id, revision, path, kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.role,
            "name": ROLE_DEFINITIONS[self.role],
            "repo_id": self.repo_id,
            "revision": self.revision,
            "path": str(self.path.resolve()),
            "kind": self.kind,
        }


def canonical_encoder(
    state: Mapping[str, torch.Tensor], *, role: str
) -> dict[str, torch.Tensor]:
    """Return the canonical encoder state as finite contiguous F32 tensors.

    Non-encoder additions such as ``proj.*`` and ``featurizer.*`` are excluded
    by construction.  The original mapping is not mutated.
    """

    selected = {key: value for key, value in state.items() if key.startswith(CANONICAL_PREFIX)}
    if not selected:
        raise ExperimentValidationError(
            f"{role} has no tensors under the canonical {CANONICAL_PREFIX!r} namespace"
        )
    out: dict[str, torch.Tensor] = {}
    for key in sorted(selected):
        value = selected[key]
        if not isinstance(value, torch.Tensor):
            raise ExperimentValidationError(f"{role}:{key} is not a torch.Tensor")
        if value.layout != torch.strided:
            raise ExperimentValidationError(f"{role}:{key} has unsupported layout {value.layout}")
        value = value.detach().to(dtype=ARITHMETIC_DTYPE).contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ExperimentValidationError(f"{role}:{key} contains NaN or infinity")
        out[key] = value
    return out


def validate_encoder_triplet(
    e: Mapping[str, torch.Tensor],
    m: Mapping[str, torch.Tensor],
    f: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    """Validate ``E``, ``M`` and ``F`` with exact keys and exact shapes."""

    states = {
        "E": canonical_encoder(e, role="E/PT_EN"),
        "M": canonical_encoder(m, role="M/PT_ML"),
        "F": canonical_encoder(f, role="F/FT_EN"),
    }
    reference_keys = set(states["E"])
    for role in ("M", "F"):
        keys = set(states[role])
        missing = sorted(reference_keys - keys)
        extra = sorted(keys - reference_keys)
        if missing or extra:
            raise ExperimentValidationError(
                f"{role}/{ROLE_DEFINITIONS[role]} tensor keys differ from E/PT_EN; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
    for key in sorted(reference_keys):
        shapes = {role: tuple(state[key].shape) for role, state in states.items()}
        if len(set(shapes.values())) != 1:
            raise ExperimentValidationError(
                f"shape mismatch for {key}: "
                + ", ".join(f"{role}={shape}" for role, shape in shapes.items())
                + "; broadcasting is forbidden"
            )
    return states


def task_vector(
    e: Mapping[str, torch.Tensor], f: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Compute the encoder-only F32 task vector ``F - E`` strictly."""

    # Reuse triplet validation without weakening the exact-key rule.
    states = validate_encoder_triplet(e, e, f)
    delta: dict[str, torch.Tensor] = {}
    for key in states["E"]:
        value = torch.sub(states["F"][key], states["E"][key])
        if value.dtype != ARITHMETIC_DTYPE or not bool(torch.isfinite(value).all()):
            raise ExperimentValidationError(f"task vector {key} is not finite F32")
        delta[key] = value.contiguous()
    return delta


def candidate(
    m: Mapping[str, torch.Tensor],
    delta: Mapping[str, torch.Tensor],
    weight: float,
) -> dict[str, torch.Tensor]:
    """Construct ``M + weight * delta`` with no implicit casts or broadcasting."""

    weight = float(weight)
    if weight not in LAMBDAS:
        raise ExperimentValidationError(
            f"lambda={weight:g} is not in the frozen sweep {list(LAMBDAS)}"
        )
    base = canonical_encoder(m, role="M/PT_ML")
    update = canonical_encoder(delta, role="delta_F")
    if set(base) != set(update):
        raise ExperimentValidationError(
            "M/PT_ML and delta_F tensor keys differ; missing or extra tensors are forbidden"
        )
    out: dict[str, torch.Tensor] = {}
    for key in base:
        if base[key].shape != update[key].shape:
            raise ExperimentValidationError(
                f"shape mismatch for {key}: M={tuple(base[key].shape)}, "
                f"delta={tuple(update[key].shape)}; broadcasting is forbidden"
            )
        value = torch.add(base[key], update[key], alpha=weight)
        if value.dtype != ARITHMETIC_DTYPE or not bool(torch.isfinite(value).all()):
            raise ExperimentValidationError(f"candidate lambda={weight:g}:{key} is not finite F32")
        out[key] = value.contiguous()
    return out


def build_candidates(
    m: Mapping[str, torch.Tensor], delta: Mapping[str, torch.Tensor]
) -> dict[float, dict[str, torch.Tensor]]:
    """Construct the complete frozen sweep, with lambda=1 as the endpoint."""

    return {weight: candidate(m, delta, weight) for weight in LAMBDAS}


def inherit_runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return an exact independent copy of the ``PT_ML`` runtime configuration."""

    inherited = copy.deepcopy(dict(config))
    if inherited != dict(config):  # defensive against unusual mapping implementations
        raise ExperimentValidationError("candidate runtime config changed while copying PT_ML")
    return inherited


def assert_runtime_config_inherited(
    candidate_config: Mapping[str, Any], pt_ml_config: Mapping[str, Any]
) -> None:
    """Reject any candidate runtime setting that differs from ``PT_ML``."""

    if dict(candidate_config) != dict(pt_ml_config):
        raise ExperimentValidationError(
            "candidate runtime configuration differs from PT_ML; candidates must inherit it exactly"
        )


def verify_declared_ancestor(
    e: CheckpointIdentity,
    f: CheckpointIdentity,
    assertion: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that F's recorded lineage names the exact pinned E checkpoint.

    Parameter values cannot prove training lineage.  This check therefore
    requires auditable external evidence and verifies that the evidence's
    ancestor identity exactly matches the ``E`` repo and pinned revision used
    by the experiment.  Tensor compatibility is checked separately.
    """

    if e.role != "E" or f.role != "F":
        raise ExperimentValidationError("ancestor verification requires E/PT_EN and F/FT_EN")
    repo_id = str(assertion.get("repo_id", "")).strip()
    revision = str(assertion.get("revision", "")).strip()
    evidence = str(assertion.get("evidence", "")).strip()
    if (repo_id, revision) != (e.repo_id, e.revision):
        raise ExperimentValidationError(
            "FT_EN ancestor assertion does not identify the exact PT_EN checkpoint: "
            f"asserted {repo_id}@{revision}, experiment uses {e.repo_id}@{e.revision}"
        )
    if not evidence or any(token in evidence.lower() for token in ("todo", "replace", "unknown")):
        raise ExperimentValidationError(
            "FT_EN ancestor assertion needs a model card, release note, or other evidence reference"
        )
    return {
        "verified": True,
        "ancestor": {"repo_id": repo_id, "revision": revision, "role": "E/PT_EN"},
        "descendant": {
            "repo_id": f.repo_id,
            "revision": f.revision,
            "role": "F/FT_EN",
        },
        "evidence": evidence,
        "verification": "pinned identity match plus exact canonical tensor compatibility",
    }


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without reading a multi-gigabyte checkpoint into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_checkpoint(
    identity: CheckpointIdentity,
    *,
    config: Mapping[str, Any],
    files: Sequence[Path],
) -> dict[str, Any]:
    """Record pinned identity, complete configs, sizes, and SHA-256 file hashes."""

    observed_revisions: dict[str, str] = {}
    config_revision = config.get("_commit_hash")
    if isinstance(config_revision, str) and config_revision.strip():
        observed_revisions["configuration._commit_hash"] = config_revision.strip()
    parts = identity.path.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            observed_revisions["huggingface_snapshot_path"] = parts[index + 1]
    for source, observed in observed_revisions.items():
        if not (
            observed.startswith(identity.revision) or identity.revision.startswith(observed)
        ):
            raise ExperimentValidationError(
                f"{identity.role} declared revision {identity.revision} disagrees with "
                f"{source}={observed}"
            )
    records = []
    for path in files:
        if not path.is_file():
            raise ExperimentValidationError(f"checkpoint file does not exist: {path}")
        records.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    value = {
        **identity.as_dict(),
        "files": records,
        "configuration": copy.deepcopy(dict(config)),
        "configuration_sha256": stable_json_sha256(config),
        "revision_verification": {
            "observed": observed_revisions,
            "artifact_binding": "sha256 file hashes",
        },
    }
    return value


def tensor_manifest(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Compact reproducibility record for an already validated encoder state."""

    canonical = canonical_encoder(state, role="tensor manifest")
    listing = [
        {"key": key, "shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch.")}
        for key, value in canonical.items()
    ]
    return {
        "namespace": CANONICAL_PREFIX,
        "count": len(listing),
        "keys_and_shapes_sha256": stable_json_sha256(listing),
        "tensors": listing,
    }


def arithmetic_summary(
    e: Mapping[str, torch.Tensor],
    m: Mapping[str, torch.Tensor],
    f: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Validate the triplet and summarize the shared arithmetic invariants."""

    states = validate_encoder_triplet(e, m, f)
    delta = task_vector(states["E"], states["F"])
    reconstruction_max = 0.0
    delta_squared = 0.0
    for key, update in delta.items():
        reconstructed = states["E"][key] + update
        reconstruction_max = max(
            reconstruction_max,
            float((reconstructed - states["F"][key]).abs().max()),
        )
        delta_squared += float(update.double().square().sum())
    zero = candidate(states["M"], delta, 0.0)
    lambda_zero_exact = all(torch.equal(zero[key], states["M"][key]) for key in zero)
    if not lambda_zero_exact:
        raise ExperimentValidationError("lambda=0 does not exactly reproduce PT_ML")
    del zero
    # Construct every endpoint now so overflow/non-finiteness cannot be deferred
    # until an expensive experiment run.
    for weight in LAMBDAS:
        candidate(states["M"], delta, weight)
    return {
        "formula": "C_lambda = M + lambda * (F - E)",
        "roles": ROLE_DEFINITIONS,
        "canonical_namespace": CANONICAL_PREFIX,
        "tensor_manifest": tensor_manifest(states["E"]),
        "arithmetic_precision": ARITHMETIC_PRECISION,
        "quantization_policy": "quantize final exported artifacts only",
        "lambdas": list(LAMBDAS),
        "primary_lambda": PRIMARY_LAMBDA,
        "lambda_zero_exact_pt_ml": lambda_zero_exact,
        "reconstruction_max_abs": reconstruction_max,
        "task_vector_l2": delta_squared ** 0.5,
    }
