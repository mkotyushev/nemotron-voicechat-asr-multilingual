"""Deterministic, content-addressed manifests for the shared experiments.

The manifest is the experimental split.  Consumers load its exact clip/take
records; they never rediscover files or reshuffle speakers at evaluation time.
Writing to an existing path is idempotent only when the content hash matches,
so an accidental rerun cannot silently move the held-out set.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .experiments import ExperimentValidationError, sha256_file, stable_json_sha256

MANIFEST_SCHEMA_VERSION = "1.0"
LIBRISPEECH_SPLITS = ("map_train", "validation", "test")


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("manifest_sha256", None)
    value["manifest_sha256"] = stable_json_sha256(value)
    return value


def _verify_digest(payload: Mapping[str, Any]) -> None:
    expected = payload.get("manifest_sha256")
    if not isinstance(expected, str):
        raise ExperimentValidationError("manifest has no manifest_sha256")
    actual_payload = dict(payload)
    actual_payload.pop("manifest_sha256", None)
    actual = stable_json_sha256(actual_payload)
    if actual != expected:
        raise ExperimentValidationError(
            f"manifest content hash mismatch: recorded {expected}, computed {actual}"
        )


def write_frozen(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write once, or accept an identical rerun; never replace a frozen split."""

    value = _with_digest(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        _verify_digest(existing)
        if existing != value:
            raise ExperimentValidationError(
                f"refusing to replace frozen manifest {path}; choose a new experiment directory"
            )
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _verify_digest(value)
    if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ExperimentValidationError(
            f"{path}: unsupported manifest schema {value.get('schema_version')!r}"
        )
    return value


def _speaker_counts(n_speakers: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if n_speakers < 3:
        raise ExperimentValidationError(
            f"speaker-disjoint train/validation/test needs at least 3 speakers, found {n_speakers}"
        )
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ExperimentValidationError("LibriSpeech split ratios must contain 3 positive values")
    total = sum(ratios)
    normalized = tuple(value / total for value in ratios)
    train = max(1, int(n_speakers * normalized[0]))
    validation = max(1, int(n_speakers * normalized[1]))
    test = n_speakers - train - validation
    while test < 1:
        if train >= validation and train > 1:
            train -= 1
        elif validation > 1:
            validation -= 1
        else:  # protected by n_speakers >= 3
            raise ExperimentValidationError("cannot allocate non-empty speaker splits")
        test = n_speakers - train - validation
    return train, validation, test


def build_librispeech_manifest(
    root: Path,
    *,
    seconds: float = 6.0,
    seed: int = 0,
    pattern: str = "**/*.flac",
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    max_clips_per_split: int | None = None,
) -> dict[str, Any]:
    """Freeze fixed crops into speaker-disjoint map-train/validation/test sets."""

    import soundfile

    from .data import SAMPLE_RATE

    root = root.resolve()
    wanted = int(round(seconds * SAMPLE_RATE))
    if wanted <= 0:
        raise ExperimentValidationError("LibriSpeech crop length must be positive")
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise ExperimentValidationError(f"no audio matching {pattern!r} under {root}")
    for path in paths:
        relative = path.relative_to(root)
        if len(relative.parts) < 3:
            raise ExperimentValidationError(
                f"{path}: expected LibriSpeech <speaker>/<chapter>/<recording> layout"
            )
        info = soundfile.info(str(path))
        if info.samplerate != SAMPLE_RATE:
            raise ExperimentValidationError(
                f"{path}: {info.samplerate} Hz, expected {SAMPLE_RATE} Hz"
            )
        if info.frames < wanted:
            continue
        speaker, chapter = relative.parts[0], relative.parts[1]
        by_speaker[speaker].append(
            {
                "path": relative.as_posix(),
                "speaker": speaker,
                "chapter": chapter,
                "offset": min(info.frames - wanted, SAMPLE_RATE // 4),
                "n_samples": wanted,
                "source_frames": int(info.frames),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    speakers = sorted(by_speaker)
    random.Random(seed).shuffle(speakers)
    n_train, n_validation, _ = _speaker_counts(len(speakers), ratios)
    assignments = {
        "map_train": set(speakers[:n_train]),
        "validation": set(speakers[n_train:n_train + n_validation]),
        "test": set(speakers[n_train + n_validation:]),
    }
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, assigned in assignments.items():
        records = [record for speaker in sorted(assigned) for record in by_speaker[speaker]]
        records.sort(key=lambda record: record["path"])
        if max_clips_per_split is not None:
            if max_clips_per_split <= 0:
                raise ExperimentValidationError("max_clips_per_split must be positive")
            random.Random(f"{seed}:{split}").shuffle(records)
            records = sorted(records[:max_clips_per_split], key=lambda record: record["path"])
        if not records:
            raise ExperimentValidationError(f"LibriSpeech {split} contains no eligible clips")
        splits[split] = records
    payload = _with_digest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset": "LibriSpeech",
            "root": str(root),
            "sample_rate": SAMPLE_RATE,
            "crop_seconds": seconds,
            "seed": seed,
            "speaker_split_ratios": dict(zip(LIBRISPEECH_SPLITS, ratios)),
            "selection": {
                "pattern": pattern,
                "max_clips_per_split": max_clips_per_split,
                "audio_hash": "sha256",
            },
            "splits": splits,
            "policy": {
                "map_fit": "map_train",
                "regularization_selection": "validation",
                "reserved_final": "test",
                "speaker_disjoint": True,
            },
        }
    )
    validate_librispeech_manifest(payload)
    return payload


def validate_librispeech_manifest(payload: Mapping[str, Any]) -> None:
    _verify_digest(payload)
    if payload.get("dataset") != "LibriSpeech":
        raise ExperimentValidationError("not a LibriSpeech manifest")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(LIBRISPEECH_SPLITS):
        raise ExperimentValidationError(
            f"LibriSpeech manifest needs exactly these splits: {LIBRISPEECH_SPLITS}"
        )
    speakers: dict[str, set[str]] = {}
    paths: set[str] = set()
    required = {"path", "speaker", "chapter", "offset", "n_samples", "sha256", "bytes"}
    for split in LIBRISPEECH_SPLITS:
        records = splits[split]
        if not isinstance(records, list) or not records:
            raise ExperimentValidationError(f"LibriSpeech {split} is empty")
        speakers[split] = set()
        for record in records:
            missing = required - set(record)
            if missing:
                raise ExperimentValidationError(f"LibriSpeech {split} record misses {sorted(missing)}")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ExperimentValidationError(f"unsafe manifest path {relative}")
            if record["path"] in paths:
                raise ExperimentValidationError(f"recording appears in multiple splits: {record['path']}")
            paths.add(record["path"])
            speakers[split].add(str(record["speaker"]))
    for left_index, left in enumerate(LIBRISPEECH_SPLITS):
        for right in LIBRISPEECH_SPLITS[left_index + 1:]:
            overlap = speakers[left] & speakers[right]
            if overlap:
                raise ExperimentValidationError(
                    f"LibriSpeech speaker leakage between {left} and {right}: {sorted(overlap)}"
                )


def _fleurs_takes(root: Path, language: str, split: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    tsv = root / "data" / language / f"{split}.tsv"
    if not tsv.is_file():
        raise ExperimentValidationError(f"missing FLEURS metadata: {tsv}")
    for row in csv.reader(tsv.open(encoding="utf-8"), delimiter="\t"):
        if len(row) < 2:
            continue
        sentence_id, filename = row[0].strip(), row[1].strip()
        if not sentence_id or not filename or filename.lower() in {"audio", "filename", "path"}:
            continue
        path = root / "x" / language / split / filename
        if path.is_file():
            out[sentence_id].append(path)
    return {key: sorted(set(paths)) for key, paths in out.items()}


def _take_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_fleurs_manifest(
    root: Path,
    *,
    languages: tuple[str, ...] = ("fr_fr", "de_de", "ru_ru"),
    reference: str = "en_us",
    split: str = "dev",
    seed: int = 0,
    max_sentences_per_language: int | None = None,
) -> dict[str, Any]:
    """Freeze parallel sentence/take choices with distinct English recordings."""

    root = root.resolve()
    languages = tuple(dict.fromkeys(languages))
    if not languages or reference in languages:
        raise ExperimentValidationError("FLEURS probe languages must be non-empty and exclude reference")
    takes = {language: _fleurs_takes(root, language, split) for language in (reference, *languages)}
    english = takes[reference]
    pairs: dict[str, list[dict[str, Any]]] = {}
    for language in languages:
        eligible = sorted(
            sentence_id
            for sentence_id in set(english) & set(takes[language])
            if len(english[sentence_id]) >= 2 and len(takes[language][sentence_id]) >= 1
        )
        if max_sentences_per_language is not None:
            if max_sentences_per_language <= 0:
                raise ExperimentValidationError("max_sentences_per_language must be positive")
            random.Random(f"{seed}:{language}").shuffle(eligible)
            eligible = sorted(eligible[:max_sentences_per_language])
        if not eligible:
            raise ExperimentValidationError(
                f"no {language} sentences have both a foreign take and two distinct {reference} takes"
            )
        rows = []
        for sentence_id in eligible:
            reference_path, query_path = english[sentence_id][:2]
            if reference_path == query_path:
                raise ExperimentValidationError(
                    f"FLEURS {sentence_id} selected the same English recording twice"
                )
            rows.append(
                {
                    "sentence_id": sentence_id,
                    "english_reference": _take_record(root, reference_path),
                    "english_query": _take_record(root, query_path),
                    "foreign_query": _take_record(root, takes[language][sentence_id][0]),
                }
            )
        pairs[language] = rows
    payload = _with_digest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset": "FLEURS",
            "root": str(root),
            "split": split,
            "reference_language": reference,
            "languages": list(languages),
            "seed": seed,
            "max_sentences_per_language": max_sentences_per_language,
            "take_selection": "lexicographically first two English takes; first foreign take",
            "pairs": pairs,
            "policy": {
                "distinct_english_reference_and_query": True,
                "regularization_selection_allowed": False,
                "final_result_tuning_allowed": False,
            },
        }
    )
    validate_fleurs_manifest(payload)
    return payload


def validate_fleurs_manifest(payload: Mapping[str, Any]) -> None:
    _verify_digest(payload)
    if payload.get("dataset") != "FLEURS":
        raise ExperimentValidationError("not a FLEURS manifest")
    languages = payload.get("languages")
    pairs = payload.get("pairs")
    if not isinstance(languages, list) or not languages or not isinstance(pairs, dict):
        raise ExperimentValidationError("FLEURS manifest has invalid languages or pairs")
    if set(languages) != set(pairs):
        raise ExperimentValidationError("FLEURS languages and pair groups differ")
    required_takes = ("english_reference", "english_query", "foreign_query")
    for language in languages:
        rows = pairs[language]
        if not isinstance(rows, list) or not rows:
            raise ExperimentValidationError(f"FLEURS {language} has no pairs")
        sentence_ids: set[str] = set()
        for row in rows:
            sentence_id = str(row.get("sentence_id", ""))
            if not sentence_id or sentence_id in sentence_ids:
                raise ExperimentValidationError(f"FLEURS {language} has duplicate/empty sentence id")
            sentence_ids.add(sentence_id)
            for key in required_takes:
                take = row.get(key)
                if not isinstance(take, dict) or not {"path", "bytes", "sha256"} <= set(take):
                    raise ExperimentValidationError(f"FLEURS {language}/{sentence_id} misses {key}")
                path = Path(str(take["path"]))
                if path.is_absolute() or ".." in path.parts:
                    raise ExperimentValidationError(f"unsafe manifest path {path}")
            if row["english_reference"]["path"] == row["english_query"]["path"]:
                raise ExperimentValidationError(
                    f"FLEURS {language}/{sentence_id} reuses the English reference as query"
                )


def resolve_take(root: Path, take: Mapping[str, Any]) -> Path:
    relative = Path(str(take["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ExperimentValidationError(f"unsafe manifest path {relative}")
    return root / relative


def verify_audio_files(payload: Mapping[str, Any], *, root: Path | None = None) -> None:
    """Verify every frozen recording still has its recorded size and SHA-256."""

    source_root = (root or Path(str(payload.get("root", "")))).resolve()
    records: list[Mapping[str, Any]] = []
    if payload.get("dataset") == "LibriSpeech":
        validate_librispeech_manifest(payload)
        records = [
            record
            for split in LIBRISPEECH_SPLITS
            for record in payload["splits"][split]
        ]
    elif payload.get("dataset") == "FLEURS":
        validate_fleurs_manifest(payload)
        records = [
            row[take]
            for language in payload["languages"]
            for row in payload["pairs"][language]
            for take in ("english_reference", "english_query", "foreign_query")
        ]
    elif payload.get("role") == "dataset_a":
        validate_dataset_a_manifest(payload)
        records = dataset_a_records(payload)
    else:
        raise ExperimentValidationError("cannot verify files for an unknown manifest dataset")
    seen: set[str] = set()
    for record in records:
        relative = str(record["path"])
        if relative in seen:
            continue
        seen.add(relative)
        path = resolve_take(source_root, record)
        if not path.is_file():
            raise ExperimentValidationError(f"frozen recording is missing: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ExperimentValidationError(f"frozen recording size changed: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise ExperimentValidationError(f"frozen recording hash changed: {path}")


def assert_model_selection_source(dataset: str, split: str) -> None:
    """One guard shared by all map fitting and regularization selection code."""

    if dataset != "LibriSpeech" or split not in {"map_train", "validation"}:
        raise ExperimentValidationError(
            "maps may be fit and regularization selected only on LibriSpeech "
            "map_train/validation; FLEURS is evaluation-only"
        )


# ---------------------------------------------------------------------------
# Dataset A -- Comparison 6's Gram collection
#
# RegMean weights each candidate by where in input space that candidate is the
# authority, so the two Gram sets must be *different* audio: English assistant
# speech for FT_EN, the same task in other languages for PT_ML.  Collecting both
# on shared audio would make the two Grams coincide and reduce Eq. 2 exactly to
# the unweighted mean, which is the one configuration that measures nothing.
#
# Both corpora arrive packed -- SLURP as a tarball of FLAC, Speech-MASSIVE as
# 48 kHz WAV inside parquet -- so ``root`` here is a prepared directory of
# 16 kHz mono clips rather than an upstream download, and the manifest records
# the archive hashes and the extraction rule alongside the per-clip SHA-256.
# ---------------------------------------------------------------------------

DATASET_A_SPLITS = ("gram", "heldout")
DATASET_A_DATASETS = ("SLURP", "Speech-MASSIVE", "CommonVoice")


def _dataset_a_payload(
    dataset: str,
    root: Path,
    *,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    seconds: float,
    seed: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from .data import SAMPLE_RATE

    if dataset not in DATASET_A_DATASETS:
        raise ExperimentValidationError(f"{dataset} is not a Dataset A corpus")
    if set(splits) != set(DATASET_A_SPLITS):
        raise ExperimentValidationError(
            f"Dataset A needs exactly the splits {DATASET_A_SPLITS}"
        )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "role": "dataset_a",
        "comparison": 6,
        "root": str(root.resolve()),
        "sample_rate": SAMPLE_RATE,
        "crop_seconds": seconds,
        "seed": seed,
        "source": dict(source),
        "selection": {**dict(selection), "audio_hash": "sha256"},
        "splits": {name: [dict(record) for record in splits[name]] for name in DATASET_A_SPLITS},
        "policy": {
            "gram_collection": "gram",
            "alpha_selection": "heldout",
            "fleurs_excluded": True,
            "librispeech_excluded": True,
        },
        **dict(extra or {}),
    }
    return _with_digest(payload)


def _validate_dataset_a(payload: Mapping[str, Any], dataset: str, required: set[str]) -> None:
    _verify_digest(payload)
    if payload.get("dataset") != dataset or payload.get("role") != "dataset_a":
        raise ExperimentValidationError(f"not a Dataset A {dataset} manifest")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(DATASET_A_SPLITS):
        raise ExperimentValidationError(f"Dataset A needs exactly the splits {DATASET_A_SPLITS}")
    seen: set[str] = set()
    utterances: set[str] = set()
    for split in DATASET_A_SPLITS:
        records = splits[split]
        if not isinstance(records, list) or not records:
            raise ExperimentValidationError(f"{dataset} {split} is empty")
        for record in records:
            missing = required - set(record)
            if missing:
                raise ExperimentValidationError(
                    f"{dataset} {split} record misses {sorted(missing)}"
                )
            relative = Path(str(record["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ExperimentValidationError(f"unsafe manifest path {relative}")
            if record["path"] in seen:
                raise ExperimentValidationError(
                    f"recording appears in more than one Dataset A split: {record['path']}"
                )
            seen.add(record["path"])
            key = f"{record.get('language', '')}:{record['utterance_id']}"
            if key in utterances:
                raise ExperimentValidationError(
                    f"{dataset} utterance appears twice: {key}"
                )
            utterances.add(key)
            if int(record["n_samples"]) <= 0 or int(record["offset"]) < 0:
                raise ExperimentValidationError(f"{dataset} record has an invalid crop")


SLURP_RECORD_FIELDS = {
    "path", "utterance_id", "slurp_id", "recording", "intent", "scenario",
    "offset", "n_samples", "source_frames", "bytes", "sha256",
}
SPEECH_MASSIVE_RECORD_FIELDS = {
    "path", "utterance_id", "language", "intent", "speaker_id",
    "offset", "n_samples", "source_frames", "bytes", "sha256",
}
# Common Voice is read speech with no task annotation, which is the point of
# the ablation: it is off-domain for the assistant task in exactly the way the
# paper's ImageNet substitution is off-domain for its tasks.
COMMON_VOICE_RECORD_FIELDS = {
    "path", "utterance_id", "language",
    "offset", "n_samples", "source_frames", "bytes", "sha256",
}


def build_slurp_manifest(
    root: Path,
    *,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    seconds: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Freeze the English assistant clips that produce ``G_F``."""

    payload = _dataset_a_payload(
        "SLURP",
        root,
        splits=splits,
        source=source,
        selection=selection,
        seconds=seconds,
        seed=seed,
        extra={"candidate": "F/FT_EN", "language": "en-US"},
    )
    validate_slurp_manifest(payload)
    return payload


def validate_slurp_manifest(payload: Mapping[str, Any]) -> None:
    _validate_dataset_a(payload, "SLURP", SLURP_RECORD_FIELDS)
    if payload.get("candidate") != "F/FT_EN":
        raise ExperimentValidationError("the SLURP Gram set belongs to F/FT_EN")


def build_speech_massive_manifest(
    root: Path,
    *,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    languages: Sequence[str],
    seconds: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Freeze the fr/de/ru clips that produce ``G_M``."""

    languages = list(dict.fromkeys(languages))
    if not languages:
        raise ExperimentValidationError("Speech-MASSIVE needs at least one language")
    payload = _dataset_a_payload(
        "Speech-MASSIVE",
        root,
        splits=splits,
        source=source,
        selection=selection,
        seconds=seconds,
        seed=seed,
        extra={"candidate": "M/PT_ML", "languages": languages},
    )
    validate_speech_massive_manifest(payload)
    return payload


def validate_speech_massive_manifest(payload: Mapping[str, Any]) -> None:
    _validate_dataset_a(payload, "Speech-MASSIVE", SPEECH_MASSIVE_RECORD_FIELDS)
    if payload.get("candidate") != "M/PT_ML":
        raise ExperimentValidationError("the Speech-MASSIVE Gram set belongs to M/PT_ML")
    languages = payload.get("languages")
    if not isinstance(languages, list) or not languages:
        raise ExperimentValidationError("Speech-MASSIVE manifest declares no languages")
    for split in DATASET_A_SPLITS:
        present = {str(record["language"]) for record in payload["splits"][split]}
        if not present <= set(languages):
            raise ExperimentValidationError(
                f"Speech-MASSIVE {split} contains undeclared languages {sorted(present - set(languages))}"
            )
        if present != set(languages):
            raise ExperimentValidationError(
                f"Speech-MASSIVE {split} omits {sorted(set(languages) - present)}; "
                "the multilingual Gram must not be one language's"
            )


def build_common_voice_manifest(
    root: Path,
    *,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    languages: Sequence[str],
    seconds: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Freeze the off-domain fr/de/ru clips for the Gram-data ablation."""

    languages = list(dict.fromkeys(languages))
    if not languages:
        raise ExperimentValidationError("Common Voice needs at least one language")
    payload = _dataset_a_payload(
        "CommonVoice",
        root,
        splits=splits,
        source=source,
        selection=selection,
        seconds=seconds,
        seed=seed,
        extra={
            "candidate": "M/PT_ML",
            "languages": languages,
            "role_note": (
                "the off-domain substitute for G_M; general read speech rather "
                "than the assistant task Speech-MASSIVE shares with SLURP"
            ),
        },
    )
    validate_common_voice_manifest(payload)
    return payload


def validate_common_voice_manifest(payload: Mapping[str, Any]) -> None:
    _validate_dataset_a(payload, "CommonVoice", COMMON_VOICE_RECORD_FIELDS)
    if payload.get("candidate") != "M/PT_ML":
        raise ExperimentValidationError("the Common Voice Gram set belongs to M/PT_ML")
    languages = payload.get("languages")
    if not isinstance(languages, list) or not languages:
        raise ExperimentValidationError("Common Voice manifest declares no languages")
    for split in DATASET_A_SPLITS:
        present = {str(record["language"]) for record in payload["splits"][split]}
        if present != set(languages):
            raise ExperimentValidationError(
                f"Common Voice {split} does not cover exactly {sorted(languages)}"
            )


def dataset_a_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    validate_dataset_a_manifest(payload)
    return [record for split in DATASET_A_SPLITS for record in payload["splits"][split]]


def validate_dataset_a_manifest(payload: Mapping[str, Any]) -> None:
    dataset = payload.get("dataset")
    if dataset == "SLURP":
        validate_slurp_manifest(payload)
    elif dataset == "Speech-MASSIVE":
        validate_speech_massive_manifest(payload)
    elif dataset == "CommonVoice":
        validate_common_voice_manifest(payload)
    else:
        raise ExperimentValidationError(f"{dataset!r} is not a Dataset A corpus")


def assert_merge_selection_source(dataset: str, split: str) -> None:
    """One guard for Comparison 6's alpha grid, mirroring the map-fitting guard.

    FLEURS is the invariant most at risk in this comparison: it is already
    manifested, already loaded by every other runner, and the obvious wrong
    choice for both Gram collection and shrinkage selection.
    """

    if dataset not in DATASET_A_DATASETS or split != "heldout":
        raise ExperimentValidationError(
            "the RegMean shrinkage may be selected only on a held-out Dataset A "
            "split; FLEURS is evaluation-only and LibriSpeech belongs to the "
            "activation-map comparisons"
        )
