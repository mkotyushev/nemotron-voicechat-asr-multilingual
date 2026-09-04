"""Write the aligned encoder as an ordinary checkpoint directory.

The alignment is a linear map on the encoder's output, and the only thing that
map ever becomes is a different 1024 -> 4480 `proj`. So the natural artifact is
not a map -- it is a checkpoint that carries its own projection:

    asr-multilingual-aligned/
        config.json            the upstream encoder config, plus provenance
        processor_config.json  the upstream one, unchanged
        model.safetensors      encoder.* unchanged, and proj.* and the featurizer
        alignment.json         the fit that produced proj
        README.md              a model card

`encoder.*` is copied from the published checkpoint byte for byte: the alignment
does not touch the encoder, and a diff against upstream should show only the
tensors that were not there before. What is new is `proj.weight` / `proj.bias`,
and the mel filterbank and STFT window, which come from the VoiceChat container
because the HF repos build them in their processor rather than shipping them.

That last part is what makes this worth doing. With those five tensors present,
`convert_asr_to_mmproj.py` needs nothing from the 12 GB container any more --
`--asr-dir` is the whole input. The alignment code, torch, and the calibration
speech become build-time-only, and serving a new encoder is a path.

Safetensors is written here rather than pulled in as a dependency for the same
reason `convert_asr_to_mmproj.SafeTensors` reads it that way: the format is a
length, a JSON header and a block of tensors, and the deployment's venv is numpy
and gguf.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

_ALIGNMENT = "alignment.json"
_BASELINE = "baseline.json"

# The published repo each config belongs to. The local directory name is not it
# -- download.sh chooses that -- and a model card gets read by someone who wants
# the real upstream, so the link is either right or absent.
UPSTREAM = {
    "nemotron3_5_asr": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "nemotron_asr_streaming": "nvidia/nemotron-speech-streaming-en-0.6b",
}


def write_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str]) -> None:
    """The safetensors container: `<u64 length><json header><tensor data>`.

    Offsets are relative to the start of the data block, and the header is
    padded with spaces so the data begins 8-byte aligned, which is what every
    reader assumes even though only the length field requires it.
    """

    header: dict[str, Any] = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    blob = bytearray()
    # Sorting makes a pass-through export byte-reproducible even when its state
    # dict came from a mapping with a different insertion order.
    for name in sorted(tensors):
        array = tensors[name]
        array = np.ascontiguousarray(array, dtype=np.float32)
        start = len(blob)
        blob += array.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [start, len(blob)],
        }

    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(encoded) % 8)) % 8
    encoded += b" " * padding

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        f.write(blob)


def export(
    output: Path,
    *,
    source: Path,
    encoder: dict[str, np.ndarray],
    proj_weight: np.ndarray,
    proj_bias: np.ndarray,
    featurizer: dict[str, np.ndarray],
    report: dict[str, Any],
) -> None:
    """Write the directory `convert_asr_to_mmproj.py --asr-dir` can consume alone."""

    output.mkdir(parents=True, exist_ok=True)
    artifact_kind = report.get("artifact_kind", "aligned_checkpoint")

    tensors: dict[str, np.ndarray] = dict(encoder)
    tensors["proj.weight"] = proj_weight
    tensors["proj.bias"] = proj_bias
    for name, value in featurizer.items():
        tensors[f"preprocessor.featurizer.{name}"] = value

    write_safetensors(
        output / "model.safetensors",
        tensors,
        {
            "derived_from": str(source),
            "artifact_kind": str(artifact_kind),
            "alignment_map": str(report.get("map")),
            "format": "pt",
        },
    )

    config = json.loads((source / "config.json").read_text())
    # Extra keys are preserved and ignored by every config loader, and this is
    # the only place a reader of the bare directory would look for the story.
    config["derived_from"] = str(source)
    if artifact_kind == "pt_ml_baseline":
        config["voicechat_baseline"] = {
            "comparison": 1,
            "projection_dim": int(proj_weight.shape[0]),
            "alignment_map": None,
            "note": (
                "Encoder weights are the upstream PT_ML tensors unchanged. "
                "proj.* and the featurizer are exact copies of the FT_EN "
                "VoiceChat source; no interface map was fitted or applied."
            ),
        }
    else:
        config["voicechat_alignment"] = {
            "map": report.get("map"),
            "projection_dim": int(proj_weight.shape[0]),
            "calibration": report.get("audio"),
            "fit_frames": report.get("fit_frames"),
            "note": (
                "Encoder weights are the upstream ones, unchanged. proj.* is "
                "VoiceChat's 1024 -> 4480 projection composed with a map fitted "
                "between this encoder's output and VoiceChat's. See alignment.json."
            ),
        }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    processor = source / "processor_config.json"
    if processor.exists():
        (output / "processor_config.json").write_text(processor.read_text())

    if artifact_kind == "pt_ml_baseline":
        (output / _BASELINE).write_text(json.dumps(report, indent=2) + "\n")
        card = baseline_model_card(
            output.name, UPSTREAM.get(config.get("model_type", "")), report
        )
    else:
        (output / _ALIGNMENT).write_text(json.dumps(report, indent=2) + "\n")
        card = model_card(output.name, UPSTREAM.get(config.get("model_type", "")), report)
    (output / "README.md").write_text(card)


def baseline_model_card(name: str, upstream: str | None, report: dict[str, Any]) -> str:
    """Describe Comparison 1 without implying that an alignment was fitted."""

    origin = (
        f"[`{upstream}`](https://huggingface.co/{upstream})" if upstream
        else "the pinned PT_ML streaming ASR checkpoint"
    )
    return f"""# {name}

Comparison 1 (`PT_ML` baseline) for the multilingual VoiceChat encoder-transfer
experiment. The encoder is an unchanged pass-through copy of {origin}. The
projection and mel-featurizer tensors are exact copies of the pinned
VoiceChat/`FT_EN` source recorded in `{report.get('shared_setup_sha256', '?')}`.

## What was changed

No alignment map was fitted or applied. The standalone ASR checkpoint does not
ship VoiceChat's 1024 -> {report.get('projection_dim', 4480)} projection or its
stored mel filterbank/window, so those tensors were attached unchanged to make
the checkpoint self-contained for conversion. `baseline.json` records the
frozen setup, source hashes, equality checks, and experiment command.

## Limits

This artifact is a research baseline, not evidence of a deployable multilingual
VoiceChat model. It contains no RNN-T decoder/joint or language-prompt projector,
and retrieval metrics are screening evidence rather than an ASR or VoiceChat
deployment evaluation.
"""


def model_card(name: str, upstream: str | None, report: dict[str, Any]) -> str:
    """A card that says what this is and, more importantly, what it is not."""

    ladder = report.get("ladder", [])
    rows = "\n".join(
        f"| `{row['map']}` | {row['r2']:+.4f} | {row['cosine_mean']:+.4f} |"
        for row in ladder
        if row.get("shippable", True)
    )
    origin = (
        f"[`{upstream}`](https://huggingface.co/{upstream})" if upstream
        else "the streaming ASR checkpoint it was built from"
    )
    return f"""# {name}

The encoder of {origin}, carrying the
1024 -> {report.get('projection_dim', 4480)} projection that lets NemotronLabs
VoiceChat 11B read it.

**This is not a speech recognition model.** The RNN-T decoder, the joint and the
language prompt projector are not here, and neither is anything that turns
frames into text on its own. It is the perception half of a speech-to-speech
model, packaged so that one build step can turn it into an `mtmd` audio
projector.

## What was changed

Nothing in the encoder. `encoder.*` is the upstream checkpoint, tensor for
tensor. What is added:

| tensor | where it came from |
|---|---|
| `proj.weight`, `proj.bias` | VoiceChat's own projection, composed with a linear map fitted between this encoder's output and VoiceChat's |
| `preprocessor.featurizer.fb`, `.window` | VoiceChat's mel filterbank and STFT window, which the upstream repo computes in its processor rather than shipping |

VoiceChat's projection was trained against VoiceChat's own encoder. Fed this
one's output untouched it produces something the language model cannot read --
it answers as if it had heard silence. The map corrects for that, and because
the encoder ends in a LayerNorm and `proj` is the very next operation, the
correction *is* a different `proj`: no extra layer, no graph change.

## How it was fitted

`{report.get('map', '?')}`, on {report.get('fit_frames', '?')} frames of English
speech from `{Path(str(report.get('audio', '?'))).name}`, scored on held-out
speakers in the {report.get('projection_dim', 4480)}-wide embedding space the
language model reads:

| map | R2 | cosine |
|---|---|---|
{rows}

The row that matters is the comparison, not the absolute value: VoiceChat reads
`nvidia/nemotron-speech-streaming-en-0.6b` with no map at all at cosine 0.59 and
answers correctly, so anything above that is enough.

## Limits

The map could only be fitted against English, because VoiceChat's encoder is the
thing being matched and it understands nothing else. Cross-lingual content
survives it -- measurably, by parallel-corpus retrieval -- but reduced, and the
speech generator downstream is English regardless, so a non-English turn can at
best come back as correct text.
"""
