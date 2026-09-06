#!/usr/bin/env python3
"""Prepare and freeze Dataset A: the Gram audio Comparison 6 merges on.

RegMean weights each candidate by where in input space that candidate is the
authority, and Eq. 2 collapses onto the unweighted mean the moment the two
Gram matrices coincide.  Dataset A is therefore two *different* corpora of the
same task: SLURP for ``F/FT_EN`` (English assistant commands) and
Speech-MASSIVE fr/de/ru for ``M/PT_ML`` (a professional localisation of the same
18 domains and 60 intents, recorded).  Because the task is held fixed and only
the language varies, the Gram difference isolates language and acoustics rather
than domain shift.  FLEURS appears nowhere: it is evaluation-only.

Both corpora arrive packed and neither is 16 kHz mono on disk, so this command
is a preparation step and not only a manifest writer.  It

  * streams ``slurp_real.tar.gz`` once, extracting the candidate recordings;
  * pulls the selected Speech-MASSIVE rows out of the published parquet shards
    and resamples their 48 kHz WAV to 16 kHz with a polyphase filter;
  * writes fixed-length 16 kHz mono FLAC crops under ``--audio-root``;
  * freezes one manifest per corpus through ``manifests.write_frozen``, with
    the archive hashes, the extraction rule and a per-clip SHA-256.

The budget comes from the widest linear input in the encoder, not from the
paper's literal sample count: ``subsampling.linear`` reads 4352 features, so
four rows per input dimension is 17,408 frames *per candidate*.  Both
candidates get the same number of fixed-length crops, so equalising the frame
count across candidates is exact rather than approximate, and every Gram is
normalised by its row count as well.

The ``heldout`` split exists for the alpha grid and nothing else.  Selecting
shrinkage on FLEURS, on LibriSpeech, or on the Gram set itself would each break
a different rule; ``manifests.assert_merge_selection_source`` is the guard.

``--common-voice`` freezes the same budget from Common Voice fr/de/ru instead,
which is the design record's in-domain-versus-out-of-domain ablation: general
read speech in the same three languages, off-domain for the assistant task.  It
is written on its own and into its own output directory, so the paired SLURP and
Speech-MASSIVE manifests stay exactly as they were frozen.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from asr_align import features, manifests
from asr_align.data import SAMPLE_RATE
from asr_align.experiments import ExperimentValidationError, sha256_file, stable_json_sha256

logger = logging.getLogger("dataset-a")

SLURP_ZENODO_RECORD = "https://zenodo.org/records/4274930"
SLURP_METADATA_REPO = "https://github.com/pswietojanski/slurp"
SPEECH_MASSIVE_REPO = "FBK-MT/Speech-MASSIVE"
SPEECH_MASSIVE_SPLIT = "validation"
SPEECH_MASSIVE_PARTITION = "dev"
SOURCE_SAMPLE_RATE = 48000
RESAMPLE = "scipy.signal.resample_poly(up=1, down=3), 48000 -> 16000 Hz"
COMMON_VOICE_RESAMPLE = "scipy.signal.resample_poly(up=1, down=<rate // 16000>) to 16000 Hz"
COMMON_VOICE_REPO = "fsicoli/common_voice_17_0"
COMMON_VOICE_SPLIT = "dev"
# SLURP records the same sentence several times; the close-talk take is the
# nearest thing in the corpus to what the deployment's microphone hears.
SLURP_RECORDING_RULE = (
    "the lexicographically first '-headset' take of each sentence, falling back "
    "to the lexicographically first take"
)
SLURP_CANDIDATE_MULTIPLE = 6
CROP_OFFSET_RULE = (
    "offset 0: these are short commands and the Gram wants the onset, unlike "
    "the LibriSpeech crops which start a quarter second in"
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _distribute(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _write_clip(path: Path, samples: np.ndarray) -> None:
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), samples.astype(np.float32), SAMPLE_RATE, format="FLAC", subtype="PCM_16")


def _record(path: Path, root: Path, *, source_frames: int, wanted: int, **extra: Any) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "offset": 0,
        "n_samples": wanted,
        "source_frames": int(source_frames),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


# ---------------------------------------------------------------------------
# SLURP -- G_F
# ---------------------------------------------------------------------------


def _slurp_candidates(metadata: Path, seed: int, wanted_clips: int) -> list[dict[str, Any]]:
    """One recording per sentence, in a seeded order, generously over-drawn.

    Duration is not in the metadata, so the crop filter can only be applied
    after extraction; the multiple is what makes one pass over the tarball
    enough.
    """

    entries: list[dict[str, Any]] = []
    with (metadata / "train.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            takes = sorted(
                str(recording["file"])
                for recording in entry.get("recordings", [])
                if str(recording.get("status", "correct")) == "correct"
            )
            if not takes:
                continue
            headset = [take for take in takes if "-headset" in take]
            entries.append(
                {
                    "slurp_id": int(entry["slurp_id"]),
                    "recording": headset[0] if headset else takes[0],
                    "intent": str(entry["intent"]),
                    "scenario": str(entry["scenario"]),
                }
            )
    if not entries:
        raise ExperimentValidationError(f"{metadata}/train.jsonl yielded no usable sentences")
    entries.sort(key=lambda entry: entry["slurp_id"])
    random.Random(seed).shuffle(entries)
    return entries[: wanted_clips * SLURP_CANDIDATE_MULTIPLE]


def _extract_slurp(
    archive: Path,
    candidates: Sequence[Mapping[str, Any]],
    audio_root: Path,
    *,
    wanted: int,
    needed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream the tarball once, keeping the ``needed`` earliest long-enough takes.

    A ``tar.gz`` is not seekable, so the whole 3.9 GB is read either way; what
    is bounded is memory.  Each candidate's order in the seeded shuffle is known
    the moment it appears, so the selection is a running ``needed``-sized set and
    the crop is all that is kept.
    """

    import soundfile

    order = {str(entry["recording"]): index for index, entry in enumerate(candidates)}
    by_recording = {str(entry["recording"]): entry for entry in candidates}
    kept: dict[str, dict[str, Any]] = {}
    seen = 0
    short = 0
    wrong_rate = 0
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = Path(member.name).name
            if not member.isfile() or name not in order:
                continue
            seen += 1
            handle = stream.extractfile(member)
            if handle is None:
                continue
            samples, rate = soundfile.read(io.BytesIO(handle.read()), dtype="float32")
            if rate != SAMPLE_RATE:
                wrong_rate += 1
                continue
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            if samples.shape[0] < wanted:
                short += 1
                continue
            kept[name] = {"crop": samples[:wanted].copy(), "source_frames": int(samples.shape[0])}
            if len(kept) > needed:
                del kept[max(kept, key=lambda key: order[key])]
    if len(kept) < needed:
        raise ExperimentValidationError(
            f"SLURP yielded {len(kept)} clips of at least {wanted / SAMPLE_RATE:.1f} s, "
            f"needed {needed}; raise SLURP_CANDIDATE_MULTIPLE"
        )
    records: list[dict[str, Any]] = []
    for name in sorted(kept, key=lambda key: order[key]):
        entry = by_recording[name]
        path = audio_root / "slurp" / f"{entry['slurp_id']:06d}-{Path(name).stem}.flac"
        _write_clip(path, kept[name]["crop"])
        records.append(
            _record(
                path,
                audio_root,
                source_frames=kept[name]["source_frames"],
                wanted=wanted,
                utterance_id=str(entry["slurp_id"]),
                slurp_id=int(entry["slurp_id"]),
                recording=name,
                intent=entry["intent"],
                scenario=entry["scenario"],
            )
        )
    diagnostics = {
        "candidates_offered": len(candidates),
        "candidates_found_in_archive": seen,
        "rejected_shorter_than_crop": short,
        "rejected_wrong_sample_rate": wrong_rate,
        "kept": len(records),
    }
    return records, diagnostics


# ---------------------------------------------------------------------------
# Speech-MASSIVE -- G_M
# ---------------------------------------------------------------------------


def _speech_massive_shards(root: Path, language: str) -> list[Path]:
    shards = sorted((root / language).glob(f"{SPEECH_MASSIVE_SPLIT}-*.parquet"))
    if not shards:
        raise ExperimentValidationError(
            f"no {SPEECH_MASSIVE_SPLIT} parquet shards for {language} under {root}"
        )
    return shards


def _speech_massive_batches(shards: Iterable[Path], columns: Sequence[str]) -> Iterator[Any]:
    import pyarrow.parquet as pq

    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=list(columns)):
            yield batch


def _resample(samples: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import resample_poly

    if rate == SAMPLE_RATE:
        return samples
    if rate % SAMPLE_RATE:
        raise ExperimentValidationError(f"cannot resample {rate} Hz to {SAMPLE_RATE} Hz")
    return resample_poly(samples, 1, rate // SAMPLE_RATE).astype(np.float32)


def _speech_massive_eligible(
    shards: Sequence[Path], excluded: set[str], wanted: int
) -> tuple[list[str], dict[str, Any]]:
    """Pass one: utterance ids long enough to crop, read from the WAV header."""

    import soundfile

    durations: dict[str, int] = {}
    short = 0
    for batch in _speech_massive_batches(shards, ("id", "audio", "partition")):
        table = batch.to_pydict()
        for identifier, audio, partition in zip(table["id"], table["audio"], table["partition"]):
            if partition != SPEECH_MASSIVE_PARTITION:
                raise ExperimentValidationError(
                    f"Speech-MASSIVE {SPEECH_MASSIVE_SPLIT} row {identifier} is partition {partition!r}"
                )
            if str(identifier) in excluded:
                continue
            info = soundfile.info(io.BytesIO(audio["bytes"]))
            frames = int(info.frames * SAMPLE_RATE / info.samplerate)
            if frames < wanted:
                short += 1
                continue
            durations[str(identifier)] = frames
    ordered = sorted(durations, key=int)
    return ordered, {
        "eligible": len(ordered),
        "rejected_shorter_than_crop": short,
        "excluded_by_gating_sample": len(excluded),
    }


def _extract_speech_massive(
    shards: Sequence[Path],
    selected: Sequence[str],
    audio_root: Path,
    language: str,
    *,
    wanted: int,
) -> list[dict[str, Any]]:
    """Pass two: pull only the selected rows and write 16 kHz crops."""

    import soundfile

    wanted_ids = set(selected)
    found: dict[str, dict[str, Any]] = {}
    for batch in _speech_massive_batches(
        shards, ("id", "audio", "intent_str", "speaker_id")
    ):
        table = batch.to_pydict()
        for identifier, audio, intent, speaker in zip(
            table["id"], table["audio"], table["intent_str"], table["speaker_id"]
        ):
            key = str(identifier)
            if key not in wanted_ids:
                continue
            samples, rate = soundfile.read(io.BytesIO(audio["bytes"]), dtype="float32")
            if rate != SOURCE_SAMPLE_RATE:
                raise ExperimentValidationError(
                    f"{language}/{key} is {rate} Hz; the recorded resample rule names "
                    f"{SOURCE_SAMPLE_RATE} Hz"
                )
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            resampled = _resample(samples, rate)
            if resampled.shape[0] < wanted:
                raise ExperimentValidationError(
                    f"{language}/{key} is shorter after resampling than its header promised"
                )
            found[key] = {
                "samples": resampled,
                "intent": str(intent),
                "speaker_id": str(speaker),
                "source_rate": int(rate),
            }
    missing = sorted(set(selected) - set(found))
    if missing:
        raise ExperimentValidationError(f"{language}: selected rows vanished from the shards: {missing[:4]}")
    records = []
    for key in selected:
        value = found[key]
        path = audio_root / "speech-massive" / language / f"{key}.flac"
        _write_clip(path, value["samples"][:wanted])
        records.append(
            _record(
                path,
                audio_root,
                source_frames=value["samples"].shape[0],
                wanted=wanted,
                utterance_id=key,
                language=language,
                intent=value["intent"],
                speaker_id=value["speaker_id"],
            )
        )
    return records


# ---------------------------------------------------------------------------
# Common Voice -- the off-domain G_M
# ---------------------------------------------------------------------------


def _extract_common_voice(
    archive: Path,
    language: str,
    audio_root: Path,
    *,
    wanted: int,
    needed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take the first ``needed`` long-enough clips out of one dev shard.

    Common Voice ships one uncompressed tar of MP3 per split, so unlike SLURP
    this can stop as soon as it has enough; the order is the shard's own, which
    is fixed by the published artifact.
    """

    import soundfile

    records: list[dict[str, Any]] = []
    seen = 0
    short = 0
    with tarfile.open(archive, "r|") as stream:
        for member in stream:
            if not member.isfile() or not member.name.lower().endswith(".mp3"):
                continue
            handle = stream.extractfile(member)
            if handle is None:
                continue
            seen += 1
            samples, rate = soundfile.read(io.BytesIO(handle.read()), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            resampled = _resample(samples, rate)
            if resampled.shape[0] < wanted:
                short += 1
                continue
            identifier = Path(member.name).stem
            path = audio_root / "common-voice" / language / f"{identifier}.flac"
            _write_clip(path, resampled[:wanted])
            records.append(
                _record(
                    path,
                    audio_root,
                    source_frames=resampled.shape[0],
                    wanted=wanted,
                    utterance_id=identifier,
                    language=language,
                )
            )
            if len(records) >= needed:
                break
    if len(records) < needed:
        raise ExperimentValidationError(
            f"Common Voice {language} yielded {len(records)} clips of at least "
            f"{wanted / SAMPLE_RATE:.1f} s, needed {needed}"
        )
    return records, {
        "clips_read": seen,
        "rejected_shorter_than_crop": short,
        "kept": len(records),
    }


def _freeze_common_voice(
    args: argparse.Namespace,
    audio_root: Path,
    *,
    wanted: int,
    gram_clips: int,
    heldout_clips: int,
) -> dict[str, Any]:
    languages = list(args.common_voice_language)
    gram_per_language = _distribute(gram_clips, len(languages))
    heldout_per_language = _distribute(heldout_clips, len(languages))
    splits: dict[str, list[dict[str, Any]]] = {"gram": [], "heldout": []}
    archives: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for index, language in enumerate(languages):
        archive = args.common_voice / f"{language}_{COMMON_VOICE_SPLIT}_0.tar"
        take = gram_per_language[index] + heldout_per_language[index]
        logger.info("streaming %s for %d clips", archive, take)
        records, per_language = _extract_common_voice(
            archive, language, audio_root, wanted=wanted, needed=take
        )
        splits["gram"].extend(records[: gram_per_language[index]])
        splits["heldout"].extend(records[gram_per_language[index]:])
        archives.append(
            {
                "path": str(archive.resolve()),
                "bytes": archive.stat().st_size,
                "sha256": sha256_file(archive),
            }
        )
        diagnostics[language] = per_language
    return manifests.build_common_voice_manifest(
        audio_root,
        splits=splits,
        source={
            "corpus": "Common Voice 17.0",
            "repository": COMMON_VOICE_REPO,
            "revision": args.common_voice_revision,
            "split": COMMON_VOICE_SPLIT,
            "files": archives,
            "resample": COMMON_VOICE_RESAMPLE,
            "labels_used": None,
            "why": (
                "the design record's in-domain versus out-of-domain Gram ablation: "
                "same three languages, general read speech instead of the assistant "
                "task, and derived from Common Voice rather than FLoRes so it cannot "
                "contaminate the FLEURS retention metric"
            ),
        },
        selection={
            "order": "the published dev shard's own member order",
            "crop_offset": CROP_OFFSET_RULE,
            "per_language_gram_clips": dict(zip(languages, gram_per_language)),
            "per_language_heldout_clips": dict(zip(languages, heldout_per_language)),
            "diagnostics": diagnostics,
        },
        languages=languages,
        seconds=args.seconds,
        seed=args.seed,
    )


# ---------------------------------------------------------------------------


def _budget(
    args: argparse.Namespace,
    audio_root: Path,
    *,
    frames_per_clip: int,
    gram_clips: int,
    heldout_clips: int,
    manifests_written: Mapping[str, Any],
    role: str,
) -> dict[str, Any]:
    value = {
        "schema_version": "1.0",
        "comparison": 6,
        "role": role,
        "command": [str(Path(sys.executable).resolve()), *sys.argv],
        "audio_root": str(audio_root),
        "crop_seconds": args.seconds,
        "frames_per_clip": frames_per_clip,
        "requested": {"gram_frames": args.gram_frames, "heldout_frames": args.heldout_frames},
        "clips": {"gram": gram_clips, "heldout": heldout_clips},
        "frames": {
            "gram_per_candidate": gram_clips * frames_per_clip,
            "heldout_per_candidate": heldout_clips * frames_per_clip,
        },
        "seconds_of_audio_per_candidate": {
            "gram": gram_clips * args.seconds,
            "heldout": heldout_clips * args.seconds,
        },
        "frames_equalized_across_candidates": True,
        "manifests": dict(manifests_written),
        "fleurs_used": False,
        "librispeech_used": False,
    }
    value["digest"] = stable_json_sha256(value)
    return value


def run(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ExperimentValidationError(
            f"refusing to write Dataset A into the non-empty {output}; choose a new directory"
        )
    paired = args.slurp_archive is not None
    if paired == (args.common_voice is not None):
        raise ExperimentValidationError(
            "freeze either the paired SLURP/Speech-MASSIVE budget or the Common "
            "Voice ablation, not both in one directory"
        )
    if paired and (args.slurp_metadata is None or args.speech_massive is None):
        raise ExperimentValidationError(
            "the paired budget needs --slurp-metadata and --speech-massive"
        )
    audio_root = args.audio_root.resolve()
    wanted = int(round(args.seconds * SAMPLE_RATE))
    frames_per_clip = features.frames_out(wanted)
    if frames_per_clip <= 0:
        raise ExperimentValidationError("the crop is shorter than one encoder frame")
    gram_clips = -(-args.gram_frames // frames_per_clip)
    heldout_clips = -(-args.heldout_frames // frames_per_clip)
    needed = gram_clips + heldout_clips
    logger.info(
        "crop %.1f s = %d frames; gram %d clips (%d frames), heldout %d clips (%d frames)",
        args.seconds, frames_per_clip, gram_clips, gram_clips * frames_per_clip,
        heldout_clips, heldout_clips * frames_per_clip,
    )

    if not paired:
        manifest = _freeze_common_voice(
            args, audio_root, wanted=wanted,
            gram_clips=gram_clips, heldout_clips=heldout_clips,
        )
        output.mkdir(parents=True, exist_ok=True)
        path = output / "common-voice.json"
        manifests.write_frozen(path, manifest)
        manifests.verify_audio_files(manifests.load_manifest(path), root=audio_root)
        _write_json(
            output / "dataset_a.json",
            _budget(
                args, audio_root, frames_per_clip=frames_per_clip,
                gram_clips=gram_clips, heldout_clips=heldout_clips,
                manifests_written={"common_voice": {
                    "path": path.name, "manifest_sha256": manifest["manifest_sha256"]}},
                role="dataset_a_ood_gram",
            ),
        )
        logger.info("Dataset A off-domain Gram frozen: %s", output)
        return output

    excluded: set[str] = set()
    exclusion_source: dict[str, Any] | None = None
    if args.massive_exclude is not None:
        sample = json.loads(args.massive_exclude.read_text(encoding="utf-8"))
        excluded = {str(entry["id"]) for entry in sample["utterances"]}
        exclusion_source = {
            "path": str(args.massive_exclude.resolve()),
            "manifest_sha256": sample.get("manifest_sha256"),
            "count": len(excluded),
            "reason": "the frozen gating-check sample; keeping Dataset B's ids clean",
        }

    # ---- SLURP ----------------------------------------------------------
    candidates = _slurp_candidates(args.slurp_metadata, args.seed, needed)
    logger.info("streaming %s for %d candidate recordings", args.slurp_archive, len(candidates))
    slurp_records, slurp_diagnostics = _extract_slurp(
        args.slurp_archive, candidates, audio_root, wanted=wanted, needed=needed
    )
    slurp_splits = {
        "gram": slurp_records[:gram_clips],
        "heldout": slurp_records[gram_clips:needed],
    }
    slurp_manifest = manifests.build_slurp_manifest(
        audio_root,
        splits=slurp_splits,
        source={
            "corpus": "SLURP",
            "partition": "train",
            "archive": {
                "url": f"{SLURP_ZENODO_RECORD}/files/slurp_real.tar.gz",
                **{
                    key: value
                    for key, value in (
                        ("path", str(args.slurp_archive.resolve())),
                        ("bytes", args.slurp_archive.stat().st_size),
                        ("sha256", sha256_file(args.slurp_archive)),
                    )
                },
            },
            "metadata": {
                "repository": SLURP_METADATA_REPO,
                "revision": args.slurp_metadata_revision,
                "files": [
                    {
                        "path": str((args.slurp_metadata / name).resolve()),
                        "bytes": (args.slurp_metadata / name).stat().st_size,
                        "sha256": sha256_file(args.slurp_metadata / name),
                    }
                    for name in ("train.jsonl",)
                ],
            },
            "reserved_for_dataset_b": "SLURP devel; this manifest draws only from train",
        },
        selection={
            "recording_rule": SLURP_RECORDING_RULE,
            "order": "seeded shuffle of the sentences, slurp_id ascending before the shuffle",
            "crop_offset": CROP_OFFSET_RULE,
            "diagnostics": slurp_diagnostics,
        },
        seconds=args.seconds,
        seed=args.seed,
    )

    # ---- Speech-MASSIVE --------------------------------------------------
    languages = list(args.language)
    gram_per_language = _distribute(gram_clips, len(languages))
    heldout_per_language = _distribute(heldout_clips, len(languages))
    massive_splits: dict[str, list[dict[str, Any]]] = {"gram": [], "heldout": []}
    shard_records: list[dict[str, Any]] = []
    massive_diagnostics: dict[str, Any] = {}
    for index, language in enumerate(languages):
        shards = _speech_massive_shards(args.speech_massive, language)
        shard_records.extend(
            {
                "path": str(shard.resolve()),
                "bytes": shard.stat().st_size,
                "sha256": sha256_file(shard),
            }
            for shard in shards
        )
        eligible, diagnostics = _speech_massive_eligible(shards, excluded, wanted)
        take = gram_per_language[index] + heldout_per_language[index]
        if len(eligible) < take:
            raise ExperimentValidationError(
                f"{language} has {len(eligible)} eligible dev clips, needs {take}"
            )
        selected = eligible[:take]
        logger.info("%s: %d eligible, taking %d", language, len(eligible), take)
        records = _extract_speech_massive(
            shards, selected, audio_root, language, wanted=wanted
        )
        massive_splits["gram"].extend(records[: gram_per_language[index]])
        massive_splits["heldout"].extend(records[gram_per_language[index]:])
        diagnostics["taken"] = take
        diagnostics["dataset_b_remainder"] = len(eligible) - take
        massive_diagnostics[language] = diagnostics
    massive_manifest = manifests.build_speech_massive_manifest(
        audio_root,
        splits=massive_splits,
        source={
            "corpus": "Speech-MASSIVE",
            "repository": SPEECH_MASSIVE_REPO,
            "revision": args.speech_massive_revision,
            "split": SPEECH_MASSIVE_SPLIT,
            "partition": SPEECH_MASSIVE_PARTITION,
            "files": shard_records,
            "resample": RESAMPLE,
            "excluded_utterances": exclusion_source,
            "reserved_for_dataset_b": (
                "the remainder of the dev partition, after this manifest's ids"
            ),
        },
        selection={
            "order": "MASSIVE utterance id ascending, the design record's 'first N'",
            "crop_offset": CROP_OFFSET_RULE,
            "per_language_gram_clips": dict(zip(languages, gram_per_language)),
            "per_language_heldout_clips": dict(zip(languages, heldout_per_language)),
            "diagnostics": massive_diagnostics,
        },
        languages=languages,
        seconds=args.seconds,
        seed=args.seed,
    )

    output.mkdir(parents=True, exist_ok=True)
    slurp_path = output / "slurp.json"
    massive_path = output / "speech-massive.json"
    manifests.write_frozen(slurp_path, slurp_manifest)
    manifests.write_frozen(massive_path, massive_manifest)
    manifests.verify_audio_files(manifests.load_manifest(slurp_path), root=audio_root)
    manifests.verify_audio_files(manifests.load_manifest(massive_path), root=audio_root)

    budget = _budget(
        args, audio_root, frames_per_clip=frames_per_clip,
        gram_clips=gram_clips, heldout_clips=heldout_clips,
        manifests_written={
            "slurp": {
                "path": slurp_path.name,
                "manifest_sha256": slurp_manifest["manifest_sha256"],
            },
            "speech_massive": {
                "path": massive_path.name,
                "manifest_sha256": massive_manifest["manifest_sha256"],
            },
        },
        role="dataset_a",
    )
    _write_json(output / "dataset_a.json", budget)
    logger.info("Dataset A frozen: %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slurp-archive", type=Path, default=None)
    parser.add_argument("--slurp-metadata", type=Path, default=None)
    parser.add_argument("--slurp-metadata-revision", default=None)
    parser.add_argument("--speech-massive", type=Path, default=None)
    parser.add_argument("--speech-massive-revision", default=None)
    parser.add_argument("--language", action="append", default=None)
    parser.add_argument("--common-voice", type=Path, default=None,
                        help="directory of Common Voice dev shards, for the off-domain ablation")
    parser.add_argument("--common-voice-revision", default=None)
    parser.add_argument("--common-voice-language", action="append", default=None)
    parser.add_argument("--massive-exclude", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--gram-frames", type=int, default=17408)
    parser.add_argument("--heldout-frames", type=int, default=4352)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.language is None:
        args.language = ["fr-FR", "de-DE", "ru-RU"]
    if args.common_voice_language is None:
        args.common_voice_language = ["fr", "de", "ru"]
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
