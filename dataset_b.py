#!/usr/bin/env python3
"""Prepare whole SLURP devel and Speech-MASSIVE dev clips for Comparison 7."""

from __future__ import annotations

import argparse
import io
import json
import logging
import tarfile
from pathlib import Path

import numpy as np
import soundfile

from asr_align import dataset_b, gating, manifests
from asr_align.experiments import ExperimentValidationError, sha256_file
from dataset_a import _resample, _speech_massive_batches

LOG = logging.getLogger("dataset-b")


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_audio(root: Path, name: str, samples: np.ndarray, rate: int) -> dict:
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = _resample(samples, rate)
    if not samples.size or not np.isfinite(samples).all():
        raise ExperimentValidationError(f"invalid audio: {name}")
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # Even on a failed/resumed preparation, never replace existing audio.
    encoded = io.BytesIO()
    soundfile.write(encoded, samples, 16000, format="FLAC", subtype="PCM_16")
    data = encoded.getvalue()
    if path.exists():
        if path.read_bytes() != data:
            raise ExperimentValidationError(f"refusing to replace Dataset B recording {path}")
    else:
        path.write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": sha256_file(path),
            "offset": 0, "n_samples": len(samples), "source_frames": len(samples)}


def prepare(args: argparse.Namespace) -> dict:
    a = {"B1": manifests.load_manifest(args.dataset_a / "slurp.json"),
         "B2": manifests.load_manifest(args.dataset_a / "speech-massive.json")}
    gate = gating.load_sample(args.gating_sample)
    excluded = dataset_b.exclusion_ids(a, gate)
    slurp_source, speech_source = a["B1"]["source"], a["B2"]["source"]
    metadata = Path(slurp_source["metadata"]["files"][0]["path"]).parent / "devel.jsonl"
    archive = Path(slurp_source["archive"]["path"])
    speech_files = [Path(f["path"]) for f in speech_source["files"]]
    native_files = {lang: args.massive / f"{gating.LOCALES[lang]}.jsonl" for lang in gating.FOREIGN_LANGUAGES}
    sources = {
        "dataset_a": {pool: manifest["manifest_sha256"] for pool, manifest in a.items()},
        "gating_sample": gate["manifest_sha256"],
        "SLURP": {"metadata": file_record(metadata), "archive": file_record(archive),
                  "metadata_revision": slurp_source["metadata"]["revision"]},
        "SpeechMASSIVE": {"revision": speech_source["revision"], "files": [file_record(p) for p in speech_files]},
        "MASSIVE": {"version": "1.1", "files": {lang: file_record(p) for lang, p in native_files.items()}},
    }
    if any(sources["SLURP"]["archive"][k] != slurp_source["archive"][k] for k in ("path", "bytes", "sha256")):
        raise ExperimentValidationError("SLURP archive differs from Dataset A source")
    if sources["SpeechMASSIVE"]["files"] != speech_source["files"]:
        raise ExperimentValidationError("Speech-MASSIVE shards differ from Dataset A sources")
    output = args.output / "dataset_b.json"
    if output.exists():
        frozen = manifests.load_manifest(output)
        if (frozen["sources"] != sources or frozen["seed"] != args.seed
                or frozen["selection"]["validation_percent"] != args.validation_percent
                or frozen["root"] != str((args.output / "audio").resolve())):
            raise ExperimentValidationError("Dataset B inputs changed; choose a new output directory")
        dataset_b.verify_audio(frozen)
        return manifests.write_frozen(output, frozen)

    wanted = {}
    for line in metadata.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        uid = str(row["slurp_id"])
        if uid in excluded["B1"]:
            continue
        takes = sorted(r["file"] for r in row["recordings"] if r.get("status", "correct") == "correct")
        if takes:
            take = next((t for t in takes if "-headset" in t), takes[0])
            wanted[take] = row
    records, found = [], set()
    root = args.output / "audio"
    LOG.info("Extracting %d full SLURP devel commands", len(wanted))
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = Path(member.name).name
            if not member.isfile() or name not in wanted:
                continue
            if name in found:
                raise ExperimentValidationError(f"duplicate SLURP archive member: {name}")
            found.add(name)
            row = wanted[name]
            samples, rate = soundfile.read(io.BytesIO(stream.extractfile(member).read()), dtype="float32")
            uid = str(row["slurp_id"])
            records.append({**write_audio(root, f"slurp/{uid}.flac", samples, rate),
                            "clip_id": f"B1/en/{uid}", "pool": "B1", "utterance_id": uid,
                            "language": "en", "partition": "devel", "recording": name,
                            "transcript": row["sentence"], "annotated_transcript": row["sentence_annotation"],
                            "intent": row["intent"], "entities": row["entities"]})
    if found != set(wanted):
        raise ExperimentValidationError(f"SLURP archive missing {len(set(wanted) - found)} selected recordings")
    for language, native_path in native_files.items():
        native = {str(r["id"]): r for r in gating._read_massive(native_path) if r["partition"] == "dev"}
        locale = gating.LOCALES[language]
        shards = [p for p in speech_files if p.parent.name == locale]
        LOG.info("Extracting full Speech-MASSIVE %s dev remainder", locale)
        for batch in _speech_massive_batches(shards, ("id", "audio", "utt", "intent_str", "locale", "partition", "speaker_id")):
            for row in batch.to_pylist():
                uid = str(row["id"])
                if uid in excluded["B2"]:
                    continue
                text = native.get(uid)
                if (text is None or row["partition"] != "dev" or row["locale"] != locale
                        or text["locale"] != locale or row["utt"] != text["utt"]
                        or row["intent_str"] != text["intent"]):
                    raise ExperimentValidationError(f"native transcript/intent mismatch: {locale}/{uid}")
                samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                records.append({**write_audio(root, f"speech-massive/{locale}/{uid}.flac", samples, rate),
                                "clip_id": f"B2/{language}/{uid}", "pool": "B2", "utterance_id": uid,
                                "language": language, "locale": locale, "partition": "dev",
                                "speaker_id": str(row["speaker_id"]), "transcript": text["utt"],
                                "annotated_transcript": text["annot_utt"], "intent": text["intent"],
                                "slot_method": text["slot_method"]})
    payload = dataset_b.build_manifest(root, records=records, sources=sources, excluded=excluded,
                                       seed=args.seed, validation_percent=args.validation_percent)
    dataset_b.verify_audio(payload)
    return manifests.write_frozen(output, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--gating-sample", type=Path, required=True)
    parser.add_argument("--massive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-percent", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = prepare(args)
    LOG.info("Frozen Dataset B %s; %s", result["manifest_sha256"],
             {k: len(v) for k, v in result["splits"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
