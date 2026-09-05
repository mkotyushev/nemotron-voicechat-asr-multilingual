"""Paired English/Russian speech-to-action evaluation for VoiceChat.

This module deliberately scores the server boundary before TTS.  Audio still
passes through the deployed streaming perception encoder and VoiceChat model,
but generated speech is discarded.  The retained outputs are the assistant
transcript and the structured tool-call event.

The first dataset made by :func:`prepare_rfcb_pilot` is a development pilot,
not a final benchmark: it uses paired RFCB/BFCL text and deterministic Silero
speech synthesis.  Its purpose is to stabilize the serving and scoring path
before that path becomes part of the immutable shared experiment setup.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import re
import subprocess
import tempfile
import time
import unicodedata
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .experiments import ExperimentValidationError, sha256_file, stable_json_sha256
from .manifests import write_frozen

MANIFEST_SCHEMA_VERSION = "voice-assistant-manifest-1.0"
RAW_SCHEMA_VERSION = "voice-assistant-raw-1.1"
RESULT_SCHEMA_VERSION = "voice-assistant-result-1.2"
LANGUAGES = ("en", "ru")
CANDIDATE_ROLES = ("comparison", "control")
PRECISIONS = ("pre_quantization", "post_quantization")
# Settings that silently change what the language model produces.  A run that
# does not pin them cannot be placed in the same table as another run.
REQUIRED_RUNTIME_SETTINGS = ("VC_REF", "TEMP")
SAMPLE_RATE = 24_000
FRAME_SAMPLES = int(SAMPLE_RATE * 0.08)

SYSTEM_PROMPT = (
    "You are a voice assistant. Always answer the user in English, even when "
    "the user speaks another language. Use a provided tool whenever it can "
    "answer the request. Do not invent a tool result: call the tool first. "
    "After receiving the tool result, state it briefly in English."
)

# These cases keep all semantic arguments numeric.  Consequently the expected
# structured call is exactly the same for English and Russian; localized string
# values cannot create a false cross-language mismatch.
PILOT_CASES: dict[str, dict[str, Any]] = {
    "simple_1": {
        "tool_output": {"status": "success", "factorial": 120},
        "response_facts": [["120"]],
    },
    "simple_3": {
        "tool_output": {"status": "success", "roots": [1, 2]},
        "response_facts": [["1"], ["2"]],
    },
    "simple_19": {
        "tool_output": {"status": "success", "greatest_common_divisor": 10},
        "response_facts": [["10"]],
    },
    "multiple_1": {
        "tool_output": {"status": "success", "area": 6, "unit": "square units"},
        "response_facts": [["6"], ["square unit", "units squared"]],
    },
    "multiple_4": {
        "tool_output": {"status": "success", "displacement": 225, "unit": "meters"},
        "response_facts": [["225"], ["meter"]],
    },
    "multiple_6": {
        "tool_output": {
            "status": "success",
            "capacitance": 8.8541878128e-9,
            "unit": "farads",
        },
        "response_facts": [["8.854"], ["farad"]],
    },
}


def _unsigned(payload: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    value = dict(payload)
    value.pop(digest_field, None)
    return value


def _verify_digest(payload: Mapping[str, Any], digest_field: str) -> None:
    recorded = payload.get(digest_field)
    if not isinstance(recorded, str):
        raise ExperimentValidationError(f"record has no {digest_field}")
    actual = stable_json_sha256(_unsigned(payload, digest_field))
    if recorded != actual:
        raise ExperimentValidationError(
            f"{digest_field} mismatch: recorded {recorded}, computed {actual}"
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ExperimentValidationError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def _by_id(
    path: Path,
    *,
    strip_ru_prefix: bool = False,
    wanted: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        identifier = str(row.get("id", ""))
        if strip_ru_prefix:
            if not identifier.startswith("RU_"):
                raise ExperimentValidationError(f"{path}: Russian id lacks RU_ prefix: {identifier}")
            identifier = identifier[3:]
        if wanted is not None and identifier not in wanted:
            continue
        if not identifier or identifier in rows:
            raise ExperimentValidationError(f"{path}: duplicate or empty id {identifier!r}")
        rows[identifier] = row
    return rows


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ExperimentValidationError(f"{path}: git revision is not a full commit: {revision}")
    return revision


def _question(row: Mapping[str, Any]) -> tuple[str, str | None]:
    try:
        message = row["question"][0][0]
        text = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ExperimentValidationError(f"invalid RFCB question: {row.get('id')}") from exc
    if message.get("role") != "user" or not isinstance(text, str) or not text.strip():
        raise ExperimentValidationError(f"invalid RFCB user question: {row.get('id')}")
    eng = message.get("eng_content")
    return text.strip(), eng.strip() if isinstance(eng, str) else None


def _normalize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _normalize_schema(item)
        for key, item in value.items()
        if not key.startswith("eng_")
    }
    if result.get("type") == "dict":
        result["type"] = "object"
    return result


def _tools(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    functions = row.get("function")
    if not isinstance(functions, list) or not functions:
        raise ExperimentValidationError(f"RFCB case {row.get('id')} has no tools")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for function in functions:
        if not isinstance(function, dict):
            raise ExperimentValidationError(f"RFCB case {row.get('id')} has an invalid tool")
        name = str(function.get("name", ""))
        parameters = _normalize_schema(function.get("parameters"))
        if not name or name in names or not isinstance(parameters, dict):
            raise ExperimentValidationError(f"RFCB case {row.get('id')} has an invalid tool schema")
        names.add(name)
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": str(function.get("description", "")),
                "parameters": parameters,
            }
        )
    return tools


def _load_silero(path: Path) -> Any:
    import torch

    if not path.is_file():
        raise ExperimentValidationError(f"Silero model is missing: {path}")
    return torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")


def _synthesize(model: Any, text: str, *, speaker: str) -> np.ndarray:
    import torch

    torch.manual_seed(0)
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=SAMPLE_RATE)
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ExperimentValidationError(f"TTS produced invalid audio for {text!r}")
    peak = float(np.max(np.abs(values)))
    if peak > 0.95:
        values = values * (0.95 / peak)
    leading = np.zeros(round(0.32 * SAMPLE_RATE), dtype=np.float32)
    trailing = np.zeros(round(0.96 * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate((leading, values, trailing))


def _tts_text(text: str, language: str) -> str:
    """Spell out notation that package TTS models do not pronounce reliably."""

    if language == "en":
        return text.replace("m/s^2", "meters per second squared")
    if language == "ru":
        return text.replace("м/с^2", "метров в секунду в квадрате")
    raise ExperimentValidationError(f"unsupported synthesis language {language!r}")


def _audio_record(path: Path, root: Path, transcript: str, synth: Mapping[str, Any]) -> dict[str, Any]:
    import soundfile

    info = soundfile.info(str(path))
    return {
        "path": path.relative_to(root).as_posix(),
        "transcript": transcript,
        "sample_rate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "seconds": float(info.frames / info.samplerate),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "synthesis": dict(synth),
    }


def _official_bfcl_audio(
    root: Path,
    *,
    revision: str,
    case_ids: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read selected AU-Harness BFCL audio embedded in pinned Parquet shards."""

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ExperimentValidationError("BFCL audio revision must be a full immutable commit")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise ExperimentValidationError(
            "official BFCL audio preparation needs the data-preparation extra"
        ) from exc

    selected: dict[str, dict[str, Any]] = {}
    files: dict[str, Any] = {}
    for category in sorted({identifier.rsplit("_", 1)[0] for identifier in case_ids}):
        path = (
            root.resolve()
            / f"BFCL_v3_{category}"
            / "test-00000-of-00001.parquet"
        )
        if not path.is_file():
            raise ExperimentValidationError(f"missing official BFCL audio shard: {path}")
        files[category] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        table = parquet.read_table(path, columns=["id", "question", "audio"])
        ids = table.column("id").to_pylist()
        if len(ids) != len(set(ids)):
            raise ExperimentValidationError(f"official BFCL audio shard has duplicate ids: {path}")
        positions = {str(identifier): index for index, identifier in enumerate(ids)}
        for identifier in case_ids:
            if identifier.rsplit("_", 1)[0] != category:
                continue
            if identifier not in positions:
                raise ExperimentValidationError(f"official BFCL audio lacks {identifier}")
            index = positions[identifier]
            question_value = table.column("question")[index].as_py()
            try:
                parsed_question = json.loads(question_value)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ExperimentValidationError(
                    f"official BFCL audio has invalid question for {identifier}"
                ) from exc
            if not isinstance(parsed_question, list) or len(parsed_question) != 1:
                raise ExperimentValidationError(
                    f"official BFCL audio question is not a single utterance for {identifier}"
                )
            audio_value = table.column("audio")[index].as_py()
            encoded = audio_value.get("bytes") if isinstance(audio_value, dict) else None
            if not isinstance(encoded, bytes) or not encoded:
                raise ExperimentValidationError(f"official BFCL audio bytes are missing for {identifier}")
            selected[identifier] = {
                "question": str(parsed_question[0]),
                "bytes": encoded,
                "source_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_bytes": len(encoded),
                "shard": category,
            }
    return selected, {
        "repository": "https://huggingface.co/datasets/ServiceNow-AI/BFCL_v3_audio",
        "revision": revision,
        "license": "Apache-2.0",
        "files": files,
    }


def prepare_rfcb_pilot(
    *,
    rfcb: Path,
    silero: Path,
    english_model: Path | None,
    russian_model: Path,
    output: Path,
    bfcl_audio: Path | None = None,
    bfcl_audio_revision: str | None = None,
    english_speaker: str = "en_0",
    russian_speaker: str = "xenia",
    case_ids: Sequence[str] = tuple(PILOT_CASES),
) -> dict[str, Any]:
    """Materialize a frozen paired RFCB/BFCL development pilot."""

    output = output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = load_manifest(manifest_path)
        verify_audio_files(manifest)
        return manifest
    if output.exists():
        raise ExperimentValidationError(
            f"refusing to populate non-empty/partial dataset directory {output}; choose a new path"
        )

    case_ids = tuple(case_ids)
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ExperimentValidationError("pilot case ids must be non-empty and unique")
    unknown = set(case_ids) - set(PILOT_CASES)
    if unknown:
        raise ExperimentValidationError(f"pilot cases have no tool result fixture: {sorted(unknown)}")
    if (english_model is None) == (bfcl_audio is None):
        raise ExperimentValidationError(
            "choose exactly one English source: --english-model or --bfcl-audio"
        )
    if bfcl_audio is not None and not bfcl_audio_revision:
        raise ExperimentValidationError("official BFCL audio needs --bfcl-audio-revision")

    data_root = rfcb.resolve() / "bfcl_eval" / "data"
    sources: dict[str, dict[str, Any]] = {}
    case_sources: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    for category in sorted({identifier.rsplit("_", 1)[0] for identifier in case_ids}):
        paths = {
            "english": data_root / f"BFCL_v3_{category}.json",
            "russian": data_root / f"BFCL_v3_RU_{category}.json",
            "ground_truth": data_root / "possible_answer" / f"BFCL_v3_{category}.json",
        }
        for key, path in paths.items():
            if not path.is_file():
                raise ExperimentValidationError(f"missing RFCB source: {path}")
            sources[f"{category}_{key}"] = {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        wanted = {
            identifier
            for identifier in case_ids
            if identifier.rsplit("_", 1)[0] == category
        }
        # RFCB currently contains duplicate ids outside this pilot.  Restrict
        # parsing to the frozen selection so unrelated source defects cannot
        # silently choose one row or prevent this manifest from being built.
        english = _by_id(paths["english"], wanted=wanted)
        russian = _by_id(paths["russian"], strip_ru_prefix=True, wanted=wanted)
        ground_truth = _by_id(paths["ground_truth"], wanted=wanted)
        for identifier in case_ids:
            if identifier.rsplit("_", 1)[0] != category:
                continue
            try:
                case_sources[identifier] = (
                    english[identifier], russian[identifier], ground_truth[identifier]
                )
            except KeyError as exc:
                raise ExperimentValidationError(
                    f"RFCB pair/ground truth is missing for {identifier}"
                ) from exc

    official_audio: dict[str, dict[str, Any]] | None = None
    official_source: dict[str, Any] | None = None
    if bfcl_audio is not None:
        official_audio, official_source = _official_bfcl_audio(
            bfcl_audio,
            revision=str(bfcl_audio_revision),
            case_ids=case_ids,
        )

    model_records = {
        "ru": {
            "path": str(russian_model.resolve()),
            "model_id": russian_model.stem,
            "speaker": russian_speaker,
            "bytes": russian_model.stat().st_size,
            "sha256": sha256_file(russian_model),
        },
    }
    if english_model is not None:
        model_records["en"] = {
            "path": str(english_model.resolve()),
            "model_id": english_model.stem,
            "speaker": english_speaker,
            "bytes": english_model.stat().st_size,
            "sha256": sha256_file(english_model),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        audio_root = temporary_root / "audio"
        audio_root.mkdir()
        models = {"ru": _load_silero(russian_model)}
        if english_model is not None:
            models["en"] = _load_silero(english_model)
        cases: list[dict[str, Any]] = []
        import soundfile

        for identifier in case_ids:
            english_row, russian_row, answer_row = case_sources[identifier]
            english_text, _ = _question(english_row)
            russian_text, translated_back = _question(russian_row)
            if translated_back != english_text:
                raise ExperimentValidationError(
                    f"RFCB {identifier} does not preserve the exact English source text"
                )
            if official_audio is not None and official_audio[identifier]["question"] != english_text:
                raise ExperimentValidationError(
                    f"official BFCL audio text differs from RFCB English for {identifier}"
                )
            ground_truth = answer_row.get("ground_truth")
            if not isinstance(ground_truth, list) or len(ground_truth) != 1:
                raise ExperimentValidationError(
                    f"pilot supports exactly one expected call for {identifier}"
                )
            audio: dict[str, dict[str, Any]] = {}
            for language, text in (("en", english_text), ("ru", russian_text)):
                if language == "en" and official_audio is not None:
                    from scipy.signal import resample_poly

                    source = official_audio[identifier]
                    values, source_rate = soundfile.read(
                        io.BytesIO(source["bytes"]), dtype="float32", always_2d=False
                    )
                    values = np.asarray(values, dtype=np.float32)
                    if values.ndim == 2:
                        values = values.mean(axis=1)
                    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                        raise ExperimentValidationError(
                            f"official BFCL audio is invalid for {identifier}"
                        )
                    divisor = math.gcd(int(source_rate), SAMPLE_RATE)
                    values = resample_poly(
                        values, SAMPLE_RATE // divisor, int(source_rate) // divisor
                    ).astype(np.float32)
                    values = np.concatenate(
                        (
                            np.zeros(round(0.32 * SAMPLE_RATE), dtype=np.float32),
                            values,
                            np.zeros(round(0.96 * SAMPLE_RATE), dtype=np.float32),
                        )
                    )
                    spoken_text = text
                    provenance = {
                        "engine": "AU-Harness BFCL-v3 recorded audio",
                        "dataset_revision": bfcl_audio_revision,
                        "source_shard": source["shard"],
                        "source_sample_rate": int(source_rate),
                        "source_bytes": source["source_bytes"],
                        "source_sha256": source["source_sha256"],
                        "resampling": "scipy.signal.resample_poly to 24000 Hz",
                        "leading_silence_ms": 320,
                        "trailing_silence_ms": 960,
                    }
                else:
                    record = model_records[language]
                    spoken_text = _tts_text(text, language)
                    values = _synthesize(
                        models[language], spoken_text, speaker=str(record["speaker"])
                    )
                    provenance = {
                        "engine": "Silero TTS",
                        "source_revision": _git_revision(silero.resolve()),
                        "model_id": record["model_id"],
                        "model_sha256": record["sha256"],
                        "speaker": record["speaker"],
                        "seed": 0,
                        "leading_silence_ms": 320,
                        "trailing_silence_ms": 960,
                    }
                path = audio_root / f"{identifier}.{language}.wav"
                soundfile.write(str(path), values, SAMPLE_RATE, subtype="PCM_16")
                audio[language] = _audio_record(
                    path,
                    temporary_root,
                    spoken_text,
                    provenance,
                )
            if audio["en"]["sha256"] == audio["ru"]["sha256"]:
                raise ExperimentValidationError(f"paired audio is identical for {identifier}")
            cases.append(
                {
                    "semantic_id": identifier,
                    "category": identifier.rsplit("_", 1)[0],
                    "source_text": {"en": english_text, "ru": russian_text},
                    "tools": _tools(english_row),
                    "ground_truth": ground_truth,
                    "tool_output": PILOT_CASES[identifier]["tool_output"],
                    "response_facts": PILOT_CASES[identifier]["response_facts"],
                    "audio": audio,
                }
            )

        payload = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "dataset": "RFCB/BFCL paired speech-to-action development pilot",
            "root": str(output),
            "split": "development",
            "languages": list(LANGUAGES),
            "sample_rate": SAMPLE_RATE,
            "system_prompt": SYSTEM_PROMPT,
            "source": {
                "rfcb": {
                    "repository": "https://github.com/sir-timio/RFCB",
                    "revision": _git_revision(rfcb.resolve()),
                    "license": "Apache-2.0 (BFCL data metadata)",
                    "files": sources,
                },
                "silero": {
                    "repository": "https://github.com/snakers4/silero-models",
                    "revision": _git_revision(silero.resolve()),
                    "models": model_records,
                },
            },
            "selection": {
                "case_ids": list(case_ids),
                "reason": (
                    "single-call numeric-argument cases; exact call is language-invariant; "
                    "includes both single-tool and multi-tool selection"
                ),
                "tuning_allowed": True,
                "final_claims_allowed": False,
            },
            "scoring": {
                "primary": "single_call_exact_accuracy",
                "separate_outputs": ["before_tts_text", "tool_call"],
                "response_target_language": "en",
                "translation_policy": (
                    "identity when output obeys the English system instruction; preserve raw "
                    "text for an externally pinned translator in the final benchmark"
                ),
            },
            "cases": cases,
        }
        if official_source is not None:
            payload["source"]["bfcl_audio"] = official_source
        written = write_frozen(temporary_root / "manifest.json", payload)
        validate_manifest(written)
        verify_audio_files(written, root=temporary_root)
        temporary_root.replace(output)
    return load_manifest(manifest_path)


def validate_manifest(payload: Mapping[str, Any]) -> None:
    _verify_digest(payload, "manifest_sha256")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ExperimentValidationError("unsupported voice-assistant manifest schema")
    if payload.get("split") != "development":
        raise ExperimentValidationError("the current voice-assistant pilot must be development-only")
    if payload.get("languages") != list(LANGUAGES):
        raise ExperimentValidationError(f"voice-assistant languages must be {LANGUAGES}")
    if payload.get("sample_rate") != SAMPLE_RATE:
        raise ExperimentValidationError(f"voice-assistant audio must be {SAMPLE_RATE} Hz")
    prompt = payload.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.isascii() or not prompt.strip():
        raise ExperimentValidationError("system prompt must be non-empty ASCII")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ExperimentValidationError("voice-assistant manifest has no cases")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ExperimentValidationError("voice-assistant case must be an object")
        identifier = str(case.get("semantic_id", ""))
        if not identifier or identifier in seen:
            raise ExperimentValidationError(f"duplicate/empty semantic id {identifier!r}")
        seen.add(identifier)
        tools = case.get("tools")
        if not isinstance(tools, list) or not tools:
            raise ExperimentValidationError(f"{identifier}: tools are missing")
        names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
        if len(names) != len(tools) or len(names) != len(set(names)):
            raise ExperimentValidationError(f"{identifier}: invalid/duplicate tool names")
        truth = case.get("ground_truth")
        if not isinstance(truth, list) or len(truth) != 1:
            raise ExperimentValidationError(f"{identifier}: exactly one ground-truth call required")
        expected_name = next(iter(truth[0]), None) if isinstance(truth[0], dict) else None
        if expected_name not in names:
            raise ExperimentValidationError(f"{identifier}: ground-truth tool is unavailable")
        audio = case.get("audio")
        if not isinstance(audio, dict) or set(audio) != set(LANGUAGES):
            raise ExperimentValidationError(f"{identifier}: needs exactly English and Russian audio")
        for language in LANGUAGES:
            record = audio[language]
            required = {
                "path", "transcript", "sample_rate", "frames", "channels",
                "bytes", "sha256", "synthesis",
            }
            if not isinstance(record, dict) or not required <= set(record):
                raise ExperimentValidationError(f"{identifier}/{language}: incomplete audio record")
            relative = Path(str(record["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ExperimentValidationError(f"{identifier}/{language}: unsafe audio path")
            if record["sample_rate"] != SAMPLE_RATE or record["channels"] != 1:
                raise ExperimentValidationError(f"{identifier}/{language}: invalid audio format")
        if audio["en"]["sha256"] == audio["ru"]["sha256"]:
            raise ExperimentValidationError(f"{identifier}: paired audio files are identical")


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value)
    return value


def verify_audio_files(payload: Mapping[str, Any], *, root: Path | None = None) -> None:
    import soundfile

    validate_manifest(payload)
    source_root = (root or Path(str(payload["root"]))).resolve()
    for case in payload["cases"]:
        for language in LANGUAGES:
            record = case["audio"][language]
            path = source_root / str(record["path"])
            if not path.is_file():
                raise ExperimentValidationError(f"frozen audio is missing: {path}")
            if path.stat().st_size != int(record["bytes"]):
                raise ExperimentValidationError(f"frozen audio size changed: {path}")
            if sha256_file(path) != record["sha256"]:
                raise ExperimentValidationError(f"frozen audio hash changed: {path}")
            info = soundfile.info(str(path))
            if (
                info.samplerate != SAMPLE_RATE
                or info.channels != 1
                or info.frames != int(record["frames"])
            ):
                raise ExperimentValidationError(f"frozen audio format changed: {path}")


def _parse_arguments(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(value, dict):
        return value, None
    if not isinstance(value, str):
        return None, "arguments are neither a JSON string nor an object"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        return None, f"arguments are invalid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, "arguments JSON is not an object"
    return parsed, None


def _schema_type_valid(value: Any, schema: Mapping[str, Any]) -> bool:
    expected = schema.get("type")
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected in {"number", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected in {"object", "dict"}:
        return isinstance(value, dict)
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        return not isinstance(item_schema, dict) or all(
            _schema_type_valid(item, item_schema) for item in value
        )
    return True


def _deep_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isfinite(float(actual)) and math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12
        )
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _deep_equal(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _deep_equal(actual[key], expected[key]) for key in actual
        )
    return type(actual) is type(expected) and actual == expected


def score_tool_calls(
    calls: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score the pilot's single expected call with typed BFCL alternatives."""

    truth = ground_truth[0]
    expected_name, expected_arguments = next(iter(truth.items()))
    schema_by_name = {str(tool["name"]): tool["parameters"] for tool in tools}
    attempted = len(calls) > 0
    exactly_one = len(calls) == 1
    call = calls[0] if calls else {}
    actual_name = str(call.get("name", ""))
    arguments, parse_error = _parse_arguments(call.get("arguments")) if calls else (None, None)
    well_formed = exactly_one and bool(actual_name) and arguments is not None
    name_correct = well_formed and actual_name == expected_name
    arguments_correct = False
    type_correct = False
    errors: list[str] = []
    if not exactly_one:
        errors.append(f"expected one call, received {len(calls)}")
    if parse_error:
        errors.append(parse_error)
    separator_variant = bool(
        well_formed
        and not name_correct
        and re.sub(r"[._]", "_", actual_name) == re.sub(r"[._]", "_", expected_name)
    )
    if well_formed and not name_correct:
        errors.append(
            f"expected tool {expected_name}, received {actual_name}"
            + (" (same name, different separator)" if separator_variant else "")
        )
    if name_correct and arguments is not None:
        parameter_schema = schema_by_name[expected_name].get("properties", {})
        unknown = set(arguments) - set(expected_arguments)
        if unknown:
            errors.append(f"unexpected arguments: {sorted(unknown)}")
        type_correct = not unknown and all(
            key in parameter_schema and _schema_type_valid(value, parameter_schema[key])
            for key, value in arguments.items()
        )
        if not type_correct:
            errors.append("one or more argument types do not match the tool schema")
        values_correct = True
        for key, choices in expected_arguments.items():
            if not isinstance(choices, list) or not choices:
                raise ExperimentValidationError(f"ground-truth choices are invalid for {expected_name}.{key}")
            omission_allowed = "" in choices
            if key not in arguments:
                if not omission_allowed:
                    values_correct = False
                    errors.append(f"missing argument: {key}")
                continue
            if not any(_deep_equal(arguments[key], choice) for choice in choices):
                values_correct = False
                errors.append(f"unexpected value for argument: {key}")
        arguments_correct = not unknown and type_correct and values_correct
    return {
        "attempted": attempted,
        "well_formed": well_formed,
        "exactly_one_call": exactly_one,
        "tool_name_correct": bool(name_correct),
        "tool_name_separator_variant": separator_variant,
        "argument_types_correct": bool(type_correct),
        "arguments_correct": bool(arguments_correct),
        "single_call_exact": bool(name_correct and arguments_correct and exactly_one),
        "expected_tool": expected_name,
        "errors": errors,
    }


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"[^\w.]+", " ", normalized, flags=re.UNICODE)
    # Keep the dot only between digits, so "8.854" stays one token while a
    # sentence-final "twenty." does not become a token of its own.
    normalized = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", normalized)
    return " ".join(normalized.split())


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}
_NUMBER_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}


def _spoken_numbers_to_digits(normalized: str) -> str:
    """Rewrite spoken English numbers as digits.

    The scored text is what the model sends to TTS, so it says "one hundred and
    twenty" where the expected fact is "120".  Without this every numeric fact
    fails even when the answer is right.
    """

    tokens = normalized.split()
    output: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token not in _NUMBER_WORDS and token not in _NUMBER_SCALES:
            output.append(token)
            index += 1
            continue
        total = 0
        current = 0
        seen = False
        while index < len(tokens):
            word = tokens[index]
            if word in _NUMBER_WORDS:
                current += _NUMBER_WORDS[word]
            elif word in _NUMBER_SCALES:
                scale = _NUMBER_SCALES[word]
                if scale == 100:
                    current = max(current, 1) * 100
                else:
                    total += max(current, 1) * scale
                    current = 0
            elif word == "and" and current and current % 100 == 0:
                # "one hundred and twenty" continues; "forty and fifty" does not.
                if index + 1 >= len(tokens) or tokens[index + 1] not in _NUMBER_WORDS:
                    break
            else:
                break
            seen = True
            index += 1
        if not seen:
            output.append(token)
            index += 1
            continue
        value = total + current
        # "eight point eight five" is one number, not three.
        if index + 1 < len(tokens) and tokens[index] == "point":
            digits = ""
            probe = index + 1
            while probe < len(tokens) and _NUMBER_WORDS.get(tokens[probe], 10) < 10:
                digits += str(_NUMBER_WORDS[tokens[probe]])
                probe += 1
            if digits:
                output.append(f"{value}.{digits}")
                index = probe
                continue
        output.append(str(value))
    return " ".join(output)


def _fact_present(alternative: str, normalized: str, spoken: str) -> bool:
    needle = _normalized_text(alternative)
    if not needle:
        return False
    if re.fullmatch(r"\d+(?:\.\d+)?", needle):
        # A bare number must be its own token: "10" is not evidence for "1".
        return needle in normalized.split() or needle in spoken.split()
    return needle in normalized or needle in spoken


def _fact_score(text: str, facts: Sequence[Sequence[str]]) -> dict[str, Any]:
    normalized = _normalized_text(text)
    spoken = _spoken_numbers_to_digits(normalized)
    groups = [
        any(_fact_present(alternative, normalized, spoken) for alternative in alternatives)
        for alternatives in facts
    ]
    return {
        "matched_groups": int(sum(groups)),
        "required_groups": len(groups),
        "all_facts_present": bool(groups and all(groups)),
        "groups": groups,
        "spoken_numbers_normalized": spoken != normalized,
    }


def _target_language_score(text: str) -> dict[str, Any]:
    cyrillic = sum("CYRILLIC" in unicodedata.name(char, "") for char in text)
    latin = sum("LATIN" in unicodedata.name(char, "") for char in text)
    return {
        "method": "Unicode script heuristic",
        "cyrillic_characters": cyrillic,
        "latin_characters": latin,
        "nonempty_and_cyrillic_free": bool(text.strip()) and cyrillic == 0,
    }


def _token_f1(left: str, right: str) -> float:
    left_tokens = _normalized_text(left).split()
    right_tokens = _normalized_text(right).split()
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _canonical_calls(calls: Sequence[Mapping[str, Any]]) -> str | None:
    canonical = []
    for call in calls:
        arguments, _ = _parse_arguments(call.get("arguments"))
        if arguments is None:
            return None
        canonical.append({"name": str(call.get("name", "")), "arguments": arguments})
    if not canonical:
        return None
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _percentile(values: np.ndarray, confidence: float) -> dict[str, float]:
    tail = (1.0 - confidence) / 2.0
    return {
        "low": float(np.quantile(values, tail)),
        "high": float(np.quantile(values, 1.0 - tail)),
    }


def _binary_summary(
    values: Sequence[bool], *, seed: int, bootstrap_samples: int, confidence: float
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, array.size, size=(bootstrap_samples, array.size))
    return {
        "rate": float(array.mean()),
        "hit_count": int(array.sum()),
        "n": int(array.size),
        "confidence_interval": {
            "level": confidence,
            "method": "bootstrap over semantic cases",
            "samples": bootstrap_samples,
            "seed": seed,
            **_percentile(array[sampled].mean(axis=1), confidence),
        },
    }


def evaluate_raw(
    manifest: Mapping[str, Any],
    raw: Mapping[str, Any],
    *,
    seed: int = 0,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    validate_manifest(manifest)
    validate_raw(raw, manifest=manifest)
    if bootstrap_samples <= 0 or not 0.0 < confidence < 1.0:
        raise ExperimentValidationError("invalid bootstrap settings")
    case_by_id = {case["semantic_id"]: case for case in manifest["cases"]}
    scored: list[dict[str, Any]] = []
    for record in raw["records"]:
        case = case_by_id[record["semantic_id"]]
        text = str(record.get("before_tts_text", ""))
        scored.append(
            {
                "semantic_id": record["semantic_id"],
                "language": record["language"],
                "transport_status": record["status"],
                "tool_call": score_tool_calls(
                    record.get("tool_calls", []), case["ground_truth"], case["tools"]
                ),
                "response_facts": _fact_score(text, case["response_facts"]),
                "target_language": _target_language_score(text),
                "before_tts_text": text,
            }
        )

    by_language: dict[str, dict[str, Any]] = {}
    metric_getters = {
        "session_transport_success_rate": lambda row: row["transport_status"]
        in {"completed", "no_response", "incomplete_response"},
        "model_response_rate": lambda row: row["transport_status"] == "completed",
        "tool_call_attempt_rate": lambda row: row["tool_call"]["attempted"],
        "well_formed_tool_call_rate": lambda row: row["tool_call"]["well_formed"],
        "tool_name_accuracy": lambda row: row["tool_call"]["tool_name_correct"],
        "argument_accuracy": lambda row: row["tool_call"]["arguments_correct"],
        "single_call_exact_accuracy": lambda row: row["tool_call"]["single_call_exact"],
        "response_fact_accuracy": lambda row: row["response_facts"]["all_facts_present"],
        "english_output_compliance_rate": lambda row: row["target_language"][
            "nonempty_and_cyrillic_free"
        ],
    }
    for language_index, language in enumerate(LANGUAGES):
        rows = [row for row in scored if row["language"] == language]
        by_language[language] = {
            name: _binary_summary(
                [bool(getter(row)) for row in rows],
                seed=seed + language_index,
                bootstrap_samples=bootstrap_samples,
                confidence=confidence,
            )
            for name, getter in metric_getters.items()
        }

    indexed = {(row["semantic_id"], row["language"]): row for row in scored}
    pairs = []
    for case in manifest["cases"]:
        identifier = case["semantic_id"]
        english = indexed[(identifier, "en")]
        russian = indexed[(identifier, "ru")]
        en_signature = _canonical_calls(
            next(
                record["tool_calls"]
                for record in raw["records"]
                if record["semantic_id"] == identifier and record["language"] == "en"
            )
        )
        ru_signature = _canonical_calls(
            next(
                record["tool_calls"]
                for record in raw["records"]
                if record["semantic_id"] == identifier and record["language"] == "ru"
            )
        )
        pairs.append(
            {
                "semantic_id": identifier,
                "english_exact": english["tool_call"]["single_call_exact"],
                "russian_exact": russian["tool_call"]["single_call_exact"],
                "same_nonempty_tool_call": en_signature is not None and en_signature == ru_signature,
                "both_exact": (
                    english["tool_call"]["single_call_exact"]
                    and russian["tool_call"]["single_call_exact"]
                ),
                "both_response_factual": (
                    english["response_facts"]["all_facts_present"]
                    and russian["response_facts"]["all_facts_present"]
                ),
                "before_tts_text_token_f1": _token_f1(
                    english["before_tts_text"], russian["before_tts_text"]
                ),
            }
        )
    en_exact = np.asarray([pair["english_exact"] for pair in pairs], dtype=np.float64)
    ru_exact = np.asarray([pair["russian_exact"] for pair in pairs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(pairs), size=(bootstrap_samples, len(pairs)))
    differences = ru_exact[sampled].mean(axis=1) - en_exact[sampled].mean(axis=1)
    conditional_denominator = int(en_exact.sum())
    conditional_hits = int(np.sum((en_exact == 1) & (ru_exact == 1)))
    paired = {
        "same_nonempty_tool_call_rate": _binary_summary(
            [pair["same_nonempty_tool_call"] for pair in pairs],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        ),
        "both_exact_tool_call_rate": _binary_summary(
            [pair["both_exact"] for pair in pairs],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        ),
        "both_response_factual_rate": _binary_summary(
            [pair["both_response_factual"] for pair in pairs],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        ),
        "russian_minus_english_exact_accuracy": {
            "difference": float(ru_exact.mean() - en_exact.mean()),
            "confidence_interval": {
                "level": confidence,
                "method": "paired bootstrap over semantic cases",
                "samples": bootstrap_samples,
                "seed": seed,
                **_percentile(differences, confidence),
            },
        },
        "russian_exact_given_english_exact": {
            "rate": (
                float(conditional_hits / conditional_denominator)
                if conditional_denominator
                else None
            ),
            "hit_count": conditional_hits,
            "n": conditional_denominator,
        },
        "before_tts_text_token_f1": {
            "mean": float(np.mean([pair["before_tts_text_token_f1"] for pair in pairs])),
            "n": len(pairs),
            "comparison_basis": (
                "direct English-text comparison because the system prompt requests English "
                "for both inputs; this is diagnostic and not the primary endpoint"
            ),
        },
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "candidate": dict(raw["candidate"]),
        "precision": raw["candidate"]["precision"],
        "manifest_sha256": manifest["manifest_sha256"],
        # Copied so one result file carries everything a cross-candidate table
        # needs to prove the rows were produced under identical conditions.
        "system_prompt_sha256": raw["system_prompt_sha256"],
        "timing": dict(raw["timing"]),
        "primary_endpoint": "single_call_exact_accuracy",
        "per_language": by_language,
        "paired": paired,
        "case_scores": scored,
        "limitations": [
            "development split; model/prompt choices may be changed after inspecting results",
            "synthetic single-speaker audio in each language",
            "six single-call numeric-argument cases; not representative of all voice assistance",
            "before-TTS text token F1 is lexical and can penalize valid paraphrases",
            "English-output compliance is a Unicode-script heuristic, not language identification",
            "fact matching reads spoken English numerals and does not score rounding: a "
            "response saying 8.85 does not satisfy a required 8.854",
        ],
    }
    validate_result(result)
    return result


def validate_raw(raw: Mapping[str, Any], *, manifest: Mapping[str, Any] | None = None) -> None:
    _verify_digest(raw, "raw_sha256")
    if raw.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ExperimentValidationError("unsupported voice-assistant raw schema")
    candidate = raw.get("candidate")
    if not isinstance(candidate, dict):
        raise ExperimentValidationError("raw result has no candidate")
    validate_candidate(candidate)
    timing = raw.get("timing")
    if not isinstance(timing, dict) or timing.get("budget_measured_from") != "end_of_streamed_audio":
        raise ExperimentValidationError(
            "raw result must record a response budget measured from the end of the audio"
        )
    if not isinstance(timing.get("response_budget_seconds"), (int, float)):
        raise ExperimentValidationError("raw result has no response budget")
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise ExperimentValidationError("raw result has no records")
    keys: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record.get("semantic_id", "")), str(record.get("language", "")))
        if not key[0] or key[1] not in LANGUAGES or key in keys:
            raise ExperimentValidationError(f"raw result has invalid/duplicate case key {key}")
        keys.add(key)
        if not isinstance(record.get("tool_calls"), list):
            raise ExperimentValidationError(f"raw record {key} has invalid tool_calls")
        if record.get("status") not in {
            "completed", "no_response", "incomplete_response", "failed"
        }:
            raise ExperimentValidationError(f"raw record {key} has invalid status")
    if manifest is not None:
        expected = {
            (case["semantic_id"], language)
            for case in manifest["cases"]
            for language in LANGUAGES
        }
        if keys != expected:
            raise ExperimentValidationError("raw records do not exactly cover the frozen manifest")
        if raw.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ExperimentValidationError("raw result used a different manifest")


def validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ExperimentValidationError("unsupported voice-assistant result schema")
    if result.get("precision") not in PRECISIONS:
        raise ExperimentValidationError("voice-assistant result has invalid precision")
    validate_candidate(result.get("candidate") or {})
    if set(result.get("per_language", {})) != set(LANGUAGES):
        raise ExperimentValidationError("voice-assistant result lacks per-language scores")
    for language in LANGUAGES:
        metrics = result["per_language"][language]
        if "single_call_exact_accuracy" not in metrics:
            raise ExperimentValidationError(f"voice-assistant result lacks {language} primary score")
    paired = result.get("paired", {})
    required = {
        "same_nonempty_tool_call_rate",
        "both_exact_tool_call_rate",
        "russian_minus_english_exact_accuracy",
        "before_tts_text_token_f1",
    }
    if not required <= set(paired):
        raise ExperimentValidationError("voice-assistant result lacks paired scores")


def _json_get(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ExperimentValidationError(f"{url} did not return a JSON object")
    return value


def server_metadata(endpoint: str, *, timeout: float = 10.0) -> dict[str, Any]:
    match = re.fullmatch(r"ws(s?)://([^/]+)(?:/.*)?", endpoint)
    if not match:
        raise ExperimentValidationError(f"unsupported WebSocket endpoint: {endpoint}")
    scheme = "https" if match.group(1) else "http"
    origin = f"{scheme}://{match.group(2)}"
    return {
        "discovery": _json_get(origin + "/", timeout),
        "health": _json_get(origin + "/v1/realtime/health", timeout),
    }


def _pcm16(path: Path) -> bytes:
    import soundfile
    from scipy.signal import resample_poly

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=False)
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 2:
        values = values.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        values = resample_poly(values, SAMPLE_RATE // divisor, int(rate) // divisor).astype(
            np.float32
        )
    values = np.clip(values, -1.0, 1.0)
    return np.rint(values * 32767.0).astype("<i2").tobytes()


async def _run_case(
    *,
    endpoint: str,
    root: Path,
    case: Mapping[str, Any],
    language: str,
    system_prompt: str,
    response_budget: float,
    pace: float,
    settle: float,
    connect_timeout: float = 30.0,
) -> dict[str, Any]:
    from websockets.asyncio.client import connect

    started = time.monotonic()
    path = root / str(case["audio"][language]["path"])
    pcm = _pcm16(path)
    # The model may only answer once it has heard the request, so the budget
    # has to start at the end of the audio.  Measuring it from the start of the
    # session instead would give a long clip less decoding time than a short
    # one -- and the Russian clips are systematically shorter than the English
    # ones, which would bias the very cross-language comparison being scored.
    frame_bytes = FRAME_SAMPLES * 2
    send_seconds = math.ceil(len(pcm) / frame_bytes) * 0.08 * max(pace, 0.0)
    transport_cap = send_seconds + response_budget + settle + 15.0
    transcript_parts: list[str] = []
    final_transcripts: list[str] = []
    calls: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    output_audio_bytes = 0
    response_done: dict[str, Any] | None = None
    first_text_ms: float | None = None
    first_tool_call_ms: float | None = None
    session_id: str | None = None
    budget_exhausted = False
    audio_ended_at: float | None = None

    async with connect(
        endpoint, max_size=16 * 1024 * 1024, open_timeout=connect_timeout
    ) as ws:
        # The outer deadline is only a transport watchdog.  The response loop
        # exits at the exact response budget below so a silent model still
        # returns its successfully observed lifecycle events instead of losing
        # them in a generic TimeoutError.
        async with asyncio.timeout(transport_cap):
            while True:
                event = json.loads(await ws.recv())
                event_counts[str(event.get("type"))] += 1
                if event.get("type") == "session.created":
                    session_id = str((event.get("session") or {}).get("id", ""))
                    break
                if event.get("type") == "error":
                    raise ExperimentValidationError(f"session creation failed: {event.get('error')}")
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {"instructions": system_prompt, "tools": case["tools"]},
                    },
                    ensure_ascii=True,
                )
            )
            while True:
                event = json.loads(await ws.recv())
                event_counts[str(event.get("type"))] += 1
                if event.get("type") == "session.updated":
                    break
                if event.get("type") == "error":
                    raise ExperimentValidationError(f"session update failed: {event.get('error')}")

            async def send_audio() -> None:
                for offset in range(0, len(pcm), frame_bytes):
                    frame = pcm[offset : offset + frame_bytes]
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(frame).decode("ascii"),
                            }
                        )
                    )
                    if pace > 0:
                        await asyncio.sleep(0.08 * pace)

            sender = asyncio.create_task(send_audio())
            last_done_at: float | None = None
            try:
                while True:
                    now = time.monotonic()
                    if audio_ended_at is None and sender.done():
                        audio_ended_at = now
                    if audio_ended_at is None:
                        deadline_remaining = send_seconds + response_budget - (now - started)
                    else:
                        deadline_remaining = response_budget - (now - audio_ended_at)
                    if deadline_remaining <= 0:
                        budget_exhausted = True
                        break
                    if audio_ended_at is not None and last_done_at is not None:
                        remaining = settle - (now - last_done_at)
                        if remaining <= 0:
                            break
                        poll = min(0.25, remaining, deadline_remaining)
                    else:
                        poll = min(0.25, deadline_remaining)
                    try:
                        raw_event = await asyncio.wait_for(ws.recv(), timeout=poll)
                    except asyncio.TimeoutError:
                        continue
                    event = json.loads(raw_event)
                    kind = str(event.get("type"))
                    event_counts[kind] += 1
                    now_ms = (time.monotonic() - started) * 1000.0
                    if kind == "response.output_audio_transcript.delta":
                        delta = str(event.get("delta", ""))
                        transcript_parts.append(delta)
                        if delta and first_text_ms is None:
                            first_text_ms = now_ms
                    elif kind == "response.output_audio_transcript.done":
                        text = str(event.get("transcript", ""))
                        if text:
                            final_transcripts.append(text)
                    elif kind == "response.function_call_arguments.done":
                        calls.append(
                            {
                                "name": str(event.get("name", "")),
                                "arguments": event.get("arguments", ""),
                                "call_id": str(event.get("call_id", "")),
                                "received_ms": now_ms,
                            }
                        )
                        if first_tool_call_ms is None:
                            first_tool_call_ms = now_ms
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "function_call_output",
                                        "call_id": str(event.get("call_id", "")),
                                        "output": json.dumps(
                                            case["tool_output"], ensure_ascii=True, sort_keys=True
                                        ),
                                    },
                                },
                                ensure_ascii=True,
                            )
                        )
                    elif kind == "response.output_audio.delta":
                        try:
                            output_audio_bytes += len(base64.b64decode(event.get("delta", "")))
                        except (ValueError, TypeError):
                            errors.append({"code": "invalid_output_audio", "event": kind})
                    elif kind == "response.done":
                        response_done = event.get("response") or {}
                        last_done_at = time.monotonic()
                    elif kind == "response.created":
                        last_done_at = None
                    elif kind == "error":
                        errors.append(dict(event.get("error") or {}))
            finally:
                await sender
            await ws.send(json.dumps({"type": "session.close"}))

    before_tts_text = final_transcripts[-1] if final_transcripts else "".join(transcript_parts)
    if response_done is not None and not errors:
        status = "completed"
    elif budget_exhausted and not calls and not before_tts_text and not errors:
        status = "no_response"
    elif budget_exhausted and not errors:
        status = "incomplete_response"
    else:
        status = "failed"
    return {
        "semantic_id": case["semantic_id"],
        "language": language,
        "audio": {
            "path": case["audio"][language]["path"],
            "sha256": case["audio"][language]["sha256"],
            "seconds": case["audio"][language]["seconds"],
        },
        "status": status,
        "before_tts_text": before_tts_text,
        "tool_calls": calls,
        "errors": errors,
        "latency_ms": {
            "first_text": first_text_ms,
            "first_tool_call": first_tool_call_ms,
            "audio_ended": (
                None if audio_ended_at is None else (audio_ended_at - started) * 1000.0
            ),
            "completed": (time.monotonic() - started) * 1000.0,
        },
        "response": response_done,
        "termination": {
            "budget_exhausted": budget_exhausted,
            "response_budget_seconds": response_budget,
            "reason": "response_budget_exhausted" if budget_exhausted else "response_completed",
        },
        "session_id": session_id,
        "event_counts": dict(sorted(event_counts.items())),
        "discarded_output_audio_bytes": output_audio_bytes,
    }


async def run_realtime(
    *,
    manifest: Mapping[str, Any],
    endpoint: str,
    candidate: Mapping[str, Any],
    expected_asr_model: str,
    response_budget: float = 30.0,
    pace: float = 1.0,
    settle: float = 2.0,
    inter_case_delay: float = 0.5,
) -> dict[str, Any]:
    """Run every frozen pair through a deployed Realtime VoiceChat server."""

    validate_manifest(manifest)
    verify_audio_files(manifest)
    if response_budget <= 0 or pace < 0 or settle < 0 or inter_case_delay < 0:
        raise ExperimentValidationError("invalid realtime timing setting")
    if candidate.get("precision") not in {"pre_quantization", "post_quantization"}:
        raise ExperimentValidationError("candidate precision must be explicit")
    validate_candidate(candidate)
    metadata = await asyncio.to_thread(server_metadata, endpoint, timeout=10.0)
    discovery = metadata["discovery"]
    health = metadata["health"]
    if discovery.get("asr_model") != expected_asr_model:
        raise ExperimentValidationError(
            f"server loaded ASR_MODEL={discovery.get('asr_model')!r}, expected {expected_asr_model!r}"
        )
    if health.get("status") != "ok" or health.get("backend_status") != "ready":
        raise ExperimentValidationError(f"VoiceChat server is not ready: {health}")

    records: list[dict[str, Any]] = []
    root = Path(str(manifest["root"])).resolve()
    for case in manifest["cases"]:
        for language in LANGUAGES:
            try:
                record = await _run_case(
                    endpoint=endpoint,
                    root=root,
                    case=case,
                    language=language,
                    system_prompt=str(manifest["system_prompt"]),
                    response_budget=response_budget,
                    pace=pace,
                    settle=settle,
                )
            except Exception as exc:
                record = {
                    "semantic_id": case["semantic_id"],
                    "language": language,
                    "audio": {
                        "path": case["audio"][language]["path"],
                        "sha256": case["audio"][language]["sha256"],
                        "seconds": case["audio"][language]["seconds"],
                    },
                    "status": "failed",
                    "before_tts_text": "",
                    "tool_calls": [],
                    "errors": [{"type": type(exc).__name__, "message": str(exc)}],
                    "latency_ms": {},
                    "response": None,
                    "termination": {
                        "budget_exhausted": False,
                        "response_budget_seconds": response_budget,
                        "reason": "exception",
                    },
                    "session_id": None,
                    "event_counts": {},
                    "discarded_output_audio_bytes": 0,
                }
            records.append(record)
            if inter_case_delay:
                await asyncio.sleep(inter_case_delay)

    raw = {
        "schema_version": RAW_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate": dict(candidate),
        "manifest_sha256": manifest["manifest_sha256"],
        "endpoint": endpoint,
        "server": metadata,
        "expected_asr_model": expected_asr_model,
        "system_prompt_sha256": stable_json_sha256(manifest["system_prompt"]),
        "timing": {
            "response_budget_seconds": response_budget,
            "budget_measured_from": "end_of_streamed_audio",
            "realtime_pace": pace,
            "settle_seconds": settle,
            "inter_case_delay_seconds": inter_case_delay,
        },
        "records": records,
    }
    raw["raw_sha256"] = stable_json_sha256(raw)
    validate_raw(raw, manifest=manifest)
    return raw


def read_runtime_environment(path: Path) -> dict[str, Any]:
    """Record the deployment environment file the pinned server was started with.

    The server's discovery endpoint reports which encoder is loaded but not the
    decoding settings around it.  Two candidates scored under different
    temperature, VAD, or runtime commits are not comparable, so the file that
    supplied them is pinned by hash and its values are recorded verbatim.
    """

    provenance = file_provenance(path)
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ExperimentValidationError(f"{path}:{number}: not a KEY=VALUE line")
        key, _, value = stripped.partition("=")
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in values:
            raise ExperimentValidationError(f"{path}:{number}: invalid or repeated key {key!r}")
        values[key] = value.strip()
    missing = [key for key in REQUIRED_RUNTIME_SETTINGS if not values.get(key)]
    if missing:
        raise ExperimentValidationError(
            f"{path}: runtime environment does not pin {missing}"
        )
    # The served encoder is what differs between rows, and it is verified
    # against the server's own discovery endpoint.  Naming it here as well
    # would let a per-row override silently contradict the recorded file.
    if "ASR_MODEL" in values or "ASR_DIR" in values:
        raise ExperimentValidationError(
            f"{path}: the shared runtime environment must not name ASR_MODEL/ASR_DIR; "
            "pass the encoder per row and let --expected-asr-model verify it"
        )
    return {
        "file": provenance,
        "values": values,
        "note": "keys absent here take the pinned compose file's defaults",
    }


def validate_candidate(candidate: Mapping[str, Any]) -> None:
    """Check the identity a scored row claims, before any audio is sent."""

    role = candidate.get("role")
    if role not in CANDIDATE_ROLES:
        raise ExperimentValidationError(f"candidate role must be one of {CANDIDATE_ROLES}")
    if not str(candidate.get("candidate_id", "")).strip():
        raise ExperimentValidationError("candidate needs a candidate_id")
    if candidate.get("precision") not in PRECISIONS:
        raise ExperimentValidationError("candidate precision must be explicit")
    if role == "comparison":
        if candidate.get("comparison") not in {1, 2, 3, 4, 5}:
            raise ExperimentValidationError("a comparison row must name comparison 1-5")
        shared_setup = candidate.get("shared_setup")
        if not isinstance(shared_setup, dict) or "sha256" not in shared_setup:
            raise ExperimentValidationError(
                "a comparison row must pin the frozen shared_setup.json it came from"
            )
    else:
        if not str(candidate.get("control", "")).strip():
            raise ExperimentValidationError("a control row must name what it controls for")
        if candidate.get("comparison") is not None:
            raise ExperimentValidationError("a control row is not one of comparisons 1-5")
    artifact = candidate.get("artifact")
    if not isinstance(artifact, dict) or "sha256" not in artifact:
        raise ExperimentValidationError("candidate must pin the served artifact by hash")
    runtime = candidate.get("runtime")
    if not isinstance(runtime, dict):
        raise ExperimentValidationError("candidate must record its runtime")
    if not re.fullmatch(r"[0-9a-f]{40}", str(runtime.get("revision", ""))):
        raise ExperimentValidationError("runtime revision must be a full immutable commit")
    environment = runtime.get("environment")
    if not isinstance(environment, dict) or "sha256" not in (environment.get("file") or {}):
        raise ExperimentValidationError(
            "candidate must pin the runtime environment file it was served with"
        )


def file_provenance(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ExperimentValidationError(f"provenance file is missing: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ExperimentValidationError(f"refusing to replace immutable result {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    """The conditions two scored rows must share to belong in one table."""

    validate_result(result)
    timing = result["timing"]
    runtime = result["candidate"]["runtime"]
    return {
        "manifest_sha256": result["manifest_sha256"],
        "system_prompt_sha256": result["system_prompt_sha256"],
        "precision": result["precision"],
        "response_budget_seconds": timing["response_budget_seconds"],
        "budget_measured_from": timing["budget_measured_from"],
        "realtime_pace": timing["realtime_pace"],
        "settle_seconds": timing["settle_seconds"],
        "runtime_revision": runtime["revision"],
        "runtime_environment_sha256": runtime["environment"]["file"]["sha256"],
    }


def _row_label(candidate: Mapping[str, Any]) -> str:
    if candidate["role"] == "control":
        return f"control: {candidate['control']}"
    return f"comparison {candidate['comparison']}"


def _sort_key(result: Mapping[str, Any]) -> tuple[int, int, str]:
    candidate = result["candidate"]
    if candidate["role"] == "control":
        return (0, 0, str(candidate["candidate_id"]))
    return (1, int(candidate["comparison"]), str(candidate["candidate_id"]))


def build_comparison(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collect scored rows into one table, refusing incomparable conditions."""

    if not results:
        raise ExperimentValidationError("a comparison needs at least one result")
    signature = run_signature(results[0])
    identifiers: set[str] = set()
    for result in results:
        other = run_signature(result)
        if other != signature:
            differing = sorted(key for key in signature if signature[key] != other.get(key))
            raise ExperimentValidationError(
                f"{result['candidate']['candidate_id']} was not scored under the same "
                f"conditions; these differ: {differing}"
            )
        identifier = str(result["candidate"]["candidate_id"])
        if identifier in identifiers:
            raise ExperimentValidationError(f"duplicate candidate in comparison: {identifier}")
        identifiers.add(identifier)

    rows = []
    for result in sorted(results, key=_sort_key):
        candidate = result["candidate"]
        english = result["per_language"]["en"]
        russian = result["per_language"]["ru"]
        paired = result["paired"]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "role": candidate["role"],
                "label": _row_label(candidate),
                "artifact_sha256": candidate["artifact"]["sha256"],
                "en": {
                    name: english[name]["rate"]
                    for name in (
                        "model_response_rate",
                        "tool_call_attempt_rate",
                        "well_formed_tool_call_rate",
                        "tool_name_accuracy",
                        "argument_accuracy",
                        "single_call_exact_accuracy",
                        "response_fact_accuracy",
                        "english_output_compliance_rate",
                    )
                },
                "ru": {
                    name: russian[name]["rate"]
                    for name in (
                        "model_response_rate",
                        "tool_call_attempt_rate",
                        "well_formed_tool_call_rate",
                        "tool_name_accuracy",
                        "argument_accuracy",
                        "single_call_exact_accuracy",
                        "response_fact_accuracy",
                        "english_output_compliance_rate",
                    )
                },
                "russian_minus_english_exact_accuracy": paired[
                    "russian_minus_english_exact_accuracy"
                ],
                "same_nonempty_tool_call_rate": paired["same_nonempty_tool_call_rate"]["rate"],
                "russian_exact_given_english_exact": paired[
                    "russian_exact_given_english_exact"
                ]["rate"],
                "n_per_language": english["single_call_exact_accuracy"]["n"],
            }
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "primary_endpoint": "single_call_exact_accuracy",
        "conditions": signature,
        "rows": rows,
    }


def render_comparison_markdown(table: Mapping[str, Any]) -> str:
    conditions = table["conditions"]
    lines = [
        "# Paired English/Russian speech-to-action tool calling",
        "",
        f"Primary endpoint: `{table['primary_endpoint']}`. "
        f"Precision stage: `{conditions['precision']}`. "
        f"N = {table['rows'][0]['n_per_language']} cases per language.",
        "",
        "All rows share one frozen manifest, system prompt, runtime commit, runtime",
        "environment, and response budget:",
        "",
        "```json",
        json.dumps(conditions, indent=2, sort_keys=True),
        "```",
        "",
        "## Primary endpoint",
        "",
        "| Row | Candidate | EN exact call | RU exact call | RU-EN paired | "
        "RU exact given EN exact | Same call |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in table["rows"]:
        difference = row["russian_minus_english_exact_accuracy"]
        interval = difference["confidence_interval"]
        conditional = row["russian_exact_given_english_exact"]
        lines.append(
            "| {label} | `{candidate}` | {en_exact:.3f} | {ru_exact:.3f} | "
            "{delta:+.3f} [{low:+.3f}, {high:+.3f}] | {conditional} | {same:.3f} |".format(
                label=row["label"],
                candidate=row["candidate_id"],
                en_exact=row["en"]["single_call_exact_accuracy"],
                ru_exact=row["ru"]["single_call_exact_accuracy"],
                delta=difference["difference"],
                low=interval["low"],
                high=interval["high"],
                conditional="n/a" if conditional is None else f"{conditional:.3f}",
                same=row["same_nonempty_tool_call_rate"],
            )
        )
    lines += [
        "",
        "`RU-EN paired` is the bootstrap difference in exact tool-call accuracy over the",
        "same semantic cases. `RU exact given EN exact` is `n/a` when no English case",
        "succeeded. `Same call` is the rate at which both languages produced the identical",
        "non-empty structured call, and stays zero when neither answered.",
        "",
        "## Where a failure happened",
        "",
        "| Row | Candidate | Lang | Responded | Tried a tool | Well formed | Tool name | "
        "Arguments | Facts | English out |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in table["rows"]:
        for language in LANGUAGES:
            scores = row[language]
            lines.append(
                "| {label} | `{candidate}` | {lang} | {responded:.3f} | {attempt:.3f} | "
                "{well_formed:.3f} | {name:.3f} | {arguments:.3f} | {facts:.3f} | "
                "{english:.3f} |".format(
                    label=row["label"],
                    candidate=row["candidate_id"],
                    lang=language,
                    responded=scores["model_response_rate"],
                    attempt=scores["tool_call_attempt_rate"],
                    well_formed=scores["well_formed_tool_call_rate"],
                    name=scores["tool_name_accuracy"],
                    arguments=scores["argument_accuracy"],
                    facts=scores["response_fact_accuracy"],
                    english=scores["english_output_compliance_rate"],
                )
            )
    lines += [
        "",
        "`Responded` is whether the model produced any turn at all, which separates an",
        "encoder the language model cannot read from one it reads but misunderstands.",
        "`Facts` scores required values in the assistant text, not wording. `English out`",
        "is a Unicode-script heuristic over the same text.",
        "",
        "This is a development pilot: it may not be used to select a candidate for a",
        "deployment claim, and its case count detects only gross differences.",
        "",
    ]
    return "\n".join(lines)
