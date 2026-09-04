#!/usr/bin/env python3
"""Materialize and validate the shared setup for comparisons 1--5.

The input is a small JSON file containing pinned checkpoint identities, local
artifact paths, auditable FT_EN lineage evidence, and dataset roots.  The output
directory contains two immutable manifests plus ``shared_setup.json`` with
checkpoint hashes/configurations and every arithmetic/evaluation policy shared
by the experiment suite.

No candidate is quantized or exported here.  Checkpoint tensors are loaded
without deployment-rounding simulation, converted to F32 for strict validation,
and discarded after the setup checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from asr_align import evaluation, manifests
from asr_align.experiments import (
    ARITHMETIC_PRECISION,
    LAMBDAS,
    ROLE_DEFINITIONS,
    SHARED_SETUP_SCHEMA_VERSION,
    CheckpointIdentity,
    ExperimentValidationError,
    arithmetic_summary,
    assert_runtime_config_inherited,
    inherit_runtime_config,
    record_checkpoint,
    sha256_file,
    stable_json_sha256,
    verify_declared_ancestor,
)
from asr_align.weights import load_asr, load_voicechat_safetensors

logger = logging.getLogger("shared-setup")


def _path(base: Path, value: Any) -> Path:
    expanded = os.path.expandvars(str(value))
    if "$" in expanded:
        raise ExperimentValidationError(f"unresolved environment variable in path {value!r}")
    path = Path(expanded).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _identity(base: Path, role: str, raw: Mapping[str, Any]) -> CheckpointIdentity:
    value = dict(raw)
    value["path"] = str(_path(base, value.get("path", "")))
    identity = CheckpointIdentity.from_dict(role, value)
    expected_kind = "voicechat_safetensors" if role == "F" else "asr"
    if identity.kind != expected_kind:
        raise ExperimentValidationError(
            f"{role}/{ROLE_DEFINITIONS[role]} must have kind {expected_kind!r}"
        )
    return identity


def _asr_files(identity: CheckpointIdentity) -> list[Path]:
    required = [identity.path / "model.safetensors", identity.path / "config.json"]
    processor = identity.path / "processor_config.json"
    if processor.is_file():
        required.append(processor)
    return required


def _voicechat_files(identity: CheckpointIdentity) -> list[Path]:
    root = identity.path if identity.path.is_dir() else identity.path.parent
    model = identity.path if identity.path.is_file() else root / "model.safetensors"
    return [model, root / "config.json"]


def _read_full_config(identity: CheckpointIdentity) -> dict[str, Any]:
    root = identity.path if identity.path.is_dir() else identity.path.parent
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def _artifact_origin(role: str, raw: Mapping[str, Any]) -> dict[str, str] | None:
    """Validate optional provenance for a locally mirrored/converted artifact.

    The checkpoint identity names the authoritative source model.  A local
    file can come from a separate immutable mirror or conversion repository;
    recording that fact avoids falsely attributing derived bytes to the source
    repository while retaining the fixed experiment role.
    """

    value = raw.get("artifact_origin")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExperimentValidationError(f"{role} artifact_origin must be an object")
    result = {
        key: str(value.get(key, "")).strip()
        for key in ("repo_id", "revision", "evidence")
    }
    if not all(result.values()):
        raise ExperimentValidationError(
            f"{role} artifact_origin needs repo_id, revision, and evidence"
        )
    revision = result["revision"].lower()
    if revision in {"main", "master", "latest", "head"} or any(
        token in revision for token in ("replace", "todo", "unknown")
    ):
        raise ExperimentValidationError(
            f"{role} artifact_origin revision {result['revision']!r} is not immutable"
        )
    if any(token in result["evidence"].lower() for token in ("replace", "todo", "unknown")):
        raise ExperimentValidationError(f"{role} artifact_origin evidence is a placeholder")
    return result


def _write_frozen_setup(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ExperimentValidationError(
                f"refusing to replace {path}; use a new output directory for a changed setup"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare(spec_path: Path, output: Path) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    base = spec_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != SHARED_SETUP_SCHEMA_VERSION:
        raise ExperimentValidationError(
            f"shared setup spec must have schema_version {SHARED_SETUP_SCHEMA_VERSION!r}"
        )
    raw_checkpoints = spec.get("checkpoints")
    if not isinstance(raw_checkpoints, dict) or set(raw_checkpoints) != set(ROLE_DEFINITIONS):
        raise ExperimentValidationError("checkpoints must define exactly E, M, and F")
    identities = {
        role: _identity(base, role, raw_checkpoints[role]) for role in ROLE_DEFINITIONS
    }
    if identities["E"].path == identities["M"].path:
        raise ExperimentValidationError("PT_EN and PT_ML resolve to the same checkpoint path")
    lineage = verify_declared_ancestor(
        identities["E"], identities["F"], raw_checkpoints["F"].get("ancestor", {})
    )

    precision = spec.get("precision", {})
    if precision.get("arithmetic") != ARITHMETIC_PRECISION:
        raise ExperimentValidationError("shared arithmetic precision must be float32")
    if precision.get("quantization_stage") != "final_artifact_only":
        raise ExperimentValidationError("quantization_stage must be final_artifact_only")

    logger.info("loading E/PT_EN without deployment rounding: %s", identities["E"].path)
    e = load_asr(identities["E"].path, mmproj_precision=False)
    logger.info("loading M/PT_ML without deployment rounding: %s", identities["M"].path)
    m = load_asr(identities["M"].path, mmproj_precision=False)
    logger.info("loading unquantized F/FT_EN into canonical F32 tensors: %s", identities["F"].path)
    f = load_voicechat_safetensors(identities["F"].path)

    logger.info("validating exact encoder keys, shapes, finiteness, and lambda sweep")
    arithmetic = arithmetic_summary(e, m, f)

    full_configs = {
        "E": _read_full_config(identities["E"]),
        "M": _read_full_config(identities["M"]),
        "F": _read_full_config(identities["F"]),
    }
    checkpoint_records = {
        "E": record_checkpoint(
            identities["E"], config=full_configs["E"], files=_asr_files(identities["E"])
        ),
        "M": record_checkpoint(
            identities["M"], config=full_configs["M"], files=_asr_files(identities["M"])
        ),
        "F": record_checkpoint(
            identities["F"], config=full_configs["F"], files=_voicechat_files(identities["F"])
        ),
    }
    for role in ROLE_DEFINITIONS:
        artifact_origin = _artifact_origin(role, raw_checkpoints[role])
        if artifact_origin is not None:
            checkpoint_records[role]["artifact_origin"] = artifact_origin

    logger.info("freezing LibriSpeech and FLEURS manifests")
    data_spec = spec.get("data", {})
    libri = data_spec.get("librispeech", {})
    fleurs = data_spec.get("fleurs", {})
    if not isinstance(libri, dict) or not isinstance(fleurs, dict):
        raise ExperimentValidationError("data must define librispeech and fleurs objects")
    seed = int(spec.get("seed", 0))
    libri_manifest = manifests.build_librispeech_manifest(
        _path(base, libri.get("root", "")),
        seconds=float(libri.get("seconds", 6.0)),
        seed=seed,
        pattern=str(libri.get("pattern", "**/*.flac")),
        ratios=tuple(float(value) for value in libri.get("speaker_split_ratios", (0.6, 0.2, 0.2))),
        max_clips_per_split=(
            None if libri.get("max_clips_per_split") is None
            else int(libri["max_clips_per_split"])
        ),
    )
    fleurs_manifest = manifests.build_fleurs_manifest(
        _path(base, fleurs.get("root", "")),
        languages=tuple(fleurs.get("languages", ("fr_fr", "de_de", "ru_ru"))),
        reference=str(fleurs.get("reference", "en_us")),
        split=str(fleurs.get("split", "dev")),
        seed=seed,
        max_sentences_per_language=(
            None if fleurs.get("max_sentences_per_language") is None
            else int(fleurs["max_sentences_per_language"])
        ),
    )
    output = output.resolve()
    libri_path = output / "manifests" / "librispeech.json"
    fleurs_path = output / "manifests" / "fleurs.json"
    manifests.write_frozen(libri_path, libri_manifest)
    manifests.write_frozen(fleurs_path, fleurs_manifest)

    # Candidate configs are not separate sources of truth.  Record that an
    # exact deep copy of M's complete config is used for every endpoint.
    candidate_config_hashes = {}
    for weight in LAMBDAS:
        inherited = inherit_runtime_config(full_configs["M"])
        assert_runtime_config_inherited(inherited, full_configs["M"])
        candidate_config_hashes[f"{weight:g}"] = stable_json_sha256(inherited)

    report = {
        "schema_version": SHARED_SETUP_SCHEMA_VERSION,
        "specification": {
            "path": str(spec_path),
            "sha256": sha256_file(spec_path),
        },
        "roles": ROLE_DEFINITIONS,
        "checkpoints": checkpoint_records,
        "lineage": lineage,
        "arithmetic": arithmetic,
        "precision": {
            "arithmetic": ARITHMETIC_PRECISION,
            "source_loading": "no simulated deployment rounding",
            "ft_en_source_note": (
                "original VoiceChat safetensors; no deployment quantization in the source"
            ),
            "quantization_stage": "final_artifact_only",
            "deployment_quantization": precision.get("deployment_quantization", "Q8_0"),
            "required_score_stages": list(evaluation.PRECISION_STAGES),
        },
        "candidate_runtime_configuration": {
            "source": "M/PT_ML",
            "source_configuration_sha256": stable_json_sha256(full_configs["M"]),
            "by_lambda": candidate_config_hashes,
            "exact_match": True,
        },
        "manifests": {
            "librispeech": {
                "path": str(libri_path),
                "sha256": libri_manifest["manifest_sha256"],
                "speaker_disjoint": True,
                "map_fit": "map_train",
                "regularization_selection": "validation",
                "reserved_final": "test",
            },
            "fleurs": {
                "path": str(fleurs_path),
                "sha256": fleurs_manifest["manifest_sha256"],
                "distinct_english_reference_and_query": True,
                "regularization_selection_allowed": False,
            },
        },
        "selection_policy": {
            "map_fit": "LibriSpeech/map_train only",
            "regularization_selection": "LibriSpeech/validation only",
            "fleurs_tuning_allowed": False,
        },
        "evaluation_contract": {
            "schema_version": evaluation.RESULT_SCHEMA_VERSION,
            "evaluator": "asr_align.evaluation.evaluate_candidate",
            "tasks": ["english_voicechat_space", *evaluation.RETRIEVAL_TASKS],
            "retrieval_metrics": [*evaluation.RETRIEVAL_METRICS, "hit_count", "n"],
            "paired_confidence_intervals_vs": "PT_ML",
            "embedding_diagnostics": "mean and norm",
            "precision_stages": list(evaluation.PRECISION_STAGES),
        },
    }
    _write_frozen_setup(output / "shared_setup.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="shared setup JSON spec")
    parser.add_argument("--output", type=Path, required=True, help="new immutable setup directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    try:
        report = prepare(args.spec, args.output)
    except ExperimentValidationError as error:
        raise SystemExit(f"shared setup rejected: {error}") from error
    output_path = args.output.resolve() / "shared_setup.json"
    logger.info("shared setup ready: %s", output_path)
    logger.info("setup file sha256: %s", sha256_file(output_path))


if __name__ == "__main__":
    main()
