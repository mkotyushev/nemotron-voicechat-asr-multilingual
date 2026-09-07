"""Frozen, native-transcript supervision pools for Comparison 7.

Audio is kept whole: cropping a command while retaining its full transcript
would create false supervision. Selection is grouped by utterance, so locales
and prompt conditions cannot cross the train/validation boundary.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import gating, manifests
from .experiments import ExperimentValidationError, stable_json_sha256

BUDGETS = (25, 50, 100)
ARMS = {
    "E1": {"encoder": "FT_EN", "initialization": "original_projection"},
    "E2": {"encoder": "PT_ML", "initialization": "comparison_3_folded_map"},
    "E3": {"encoder": "regmean-plus-plus", "initialization": "original_projection"},
    "E4": {"encoder": "simple-average", "initialization": "original_projection"},
}


def exclusion_ids(dataset_a: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, list[str]]:
    """Exclude all A splits and all locales of a selected Speech-MASSIVE id."""
    excluded = {"B1": set(), "B2": {str(row["id"]) for row in gate["utterances"]}}
    for pool, manifest in dataset_a.items():
        if pool not in excluded:
            raise ExperimentValidationError(f"unknown Dataset B pool: {pool}")
        manifests.validate_dataset_a_manifest(manifest)
        expected = "SLURP" if pool == "B1" else "Speech-MASSIVE"
        if manifest["dataset"] != expected:
            raise ExperimentValidationError(f"{pool} exclusion source must be {expected}")
        excluded[pool].update(str(row["utterance_id"]) for row in manifests.dataset_a_records(manifest))
    return {pool: sorted(ids) for pool, ids in excluded.items()}


def split_records(records: Sequence[Mapping[str, Any]], *, seed: int, validation_percent: int = 10):
    if not 0 < validation_percent < 100:
        raise ExperimentValidationError("validation_percent must be between 0 and 100")
    groups: dict[str, set[str]] = defaultdict(set)
    for row in records:
        groups[str(row["pool"])].add(str(row["utterance_id"]))
    held_out = set()
    for pool, ids in groups.items():
        if len(ids) < 2:
            raise ExperimentValidationError(f"{pool} needs at least two distinct utterances")
        ordered = sorted(ids, key=lambda uid: stable_json_sha256([seed, pool, uid]))
        count = max(1, len(ordered) * validation_percent // 100)
        held_out.update((pool, uid) for uid in ordered[:count])
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for row in sorted(records, key=lambda r: str(r["clip_id"])):
        split = "validation" if (row["pool"], str(row["utterance_id"])) in held_out else "train"
        splits[split].append(dict(row))
    return splits


def budget_clip_ids(manifest: Mapping[str, Any], percentage: int) -> list[str]:
    """Nested, stratified unique-audio draws shared by all arms and prompts."""
    if percentage not in BUDGETS:
        raise ExperimentValidationError(f"budget must be one of {BUDGETS}")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifest["splits"]["train"]:
        groups[row["pool"], row["language"]].append(row)
    selected = []
    for (pool, language), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: stable_json_sha256(
            [manifest["seed"], pool, language, r["clip_id"]]))
        selected.extend(row["clip_id"] for row in ordered[:max(1, len(rows) * percentage // 100)])
    return selected


def build_manifest(root: Path, *, records: Sequence[Mapping[str, Any]], sources: Mapping[str, Any],
                   excluded: Mapping[str, Sequence[str]], seed: int = 0,
                   validation_percent: int = 10) -> dict[str, Any]:
    payload = {
        "schema_version": manifests.MANIFEST_SCHEMA_VERSION,
        "dataset": "InterfaceDatasetB", "role": "dataset_b", "comparison": 7,
        "root": str(root.resolve()), "sample_rate": 16000, "seed": seed,
        "sources": dict(sources), "excluded_ids": dict(excluded),
        "system_prompts": dict(gating.SYSTEM_PROMPTS),
        "teacher_paths": {"B1": "original_voicechat_audio", "B2": "text_chat_native_transcript"},
        "selection": {
            "validation_percent": validation_percent,
            "split_unit": "pool and utterance_id; all locales stay together",
            "budget_unit": "unique audio, before target gating; same clips for every arm",
            "audio": "whole recording, mono PCM16 FLAC at 16 kHz; no command truncation",
            "reserved_test_used": False,
        },
        "splits": split_records(records, seed=seed, validation_percent=validation_percent),
    }
    payload["budgets"] = {str(p): budget_clip_ids(payload, p) for p in BUDGETS}
    validate_manifest(payload)
    return manifests._with_digest(payload)


def validate_manifest(payload: Mapping[str, Any]) -> None:
    if "manifest_sha256" in payload:
        manifests._verify_digest(payload)
    if payload.get("dataset") != "InterfaceDatasetB" or payload.get("comparison") != 7:
        raise ExperimentValidationError("not a Comparison 7 Dataset B manifest")
    if payload.get("system_prompts") != gating.SYSTEM_PROMPTS:
        raise ExperimentValidationError("Dataset B prompt conditions changed")
    if payload.get("sample_rate") != 16000 or set(payload.get("splits", {})) != {"train", "validation"}:
        raise ExperimentValidationError("invalid Dataset B sampling rate or splits")
    seen_clips, seen_paths, group_splits = set(), set(), {}
    for split, rows in payload["splits"].items():
        cells = set()
        for row in rows:
            pool, uid = row["pool"], str(row["utterance_id"])
            if pool not in {"B1", "B2"} or uid in payload["excluded_ids"][pool]:
                raise ExperimentValidationError(f"excluded or invalid Dataset B utterance: {pool}/{uid}")
            if row["clip_id"] in seen_clips or row["path"] in seen_paths:
                raise ExperimentValidationError("duplicate Dataset B clip or recording path")
            seen_clips.add(row["clip_id"]); seen_paths.add(row["path"])
            if group_splits.setdefault((pool, uid), split) != split:
                raise ExperimentValidationError("Dataset B utterance leaks across splits")
            manifests.resolve_take(Path(payload["root"]), row)
            if row["offset"] != 0 or row["n_samples"] != row["source_frames"] or row["n_samples"] <= 0:
                raise ExperimentValidationError("Dataset B requires full, nonempty recordings")
            if not row.get("transcript") or not row.get("intent") or not row.get("sha256") or row["bytes"] <= 0:
                raise ExperimentValidationError("incomplete Dataset B content or audio provenance")
            if pool == "B1":
                if row["language"] != "en" or row["partition"] != "devel":
                    raise ExperimentValidationError("B1 must use SLURP devel English")
            elif (row["language"] not in gating.FOREIGN_LANGUAGES or row["partition"] != "dev"
                  or row["locale"] != gating.LOCALES[row["language"]]
                  or "slot_method" not in row or not isinstance(row["slot_method"], list)):
                raise ExperimentValidationError("B2 must carry native MASSIVE dev text and slot methods")
            cells.add((pool, row["language"]))
        if cells != {("B1", "en"), ("B2", "fr"), ("B2", "de"), ("B2", "ru")}:
            raise ExperimentValidationError(f"{split} must contain B1 English and B2 fr/de/ru")
    if payload.get("budgets") != {str(p): budget_clip_ids(payload, p) for p in BUDGETS}:
        raise ExperimentValidationError("Dataset B budget draws do not match their frozen rule")


def verify_audio(manifest: Mapping[str, Any]) -> None:
    from .experiments import sha256_file
    import soundfile

    validate_manifest(manifest)
    for rows in manifest["splits"].values():
        for row in rows:
            path = manifests.resolve_take(Path(manifest["root"]), row)
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                raise ExperimentValidationError(f"Dataset B audio changed: {path}")
            info = soundfile.info(path)
            if info.samplerate != 16000 or info.channels != 1 or info.frames != row["n_samples"]:
                raise ExperimentValidationError(f"Dataset B audio shape changed: {path}")
