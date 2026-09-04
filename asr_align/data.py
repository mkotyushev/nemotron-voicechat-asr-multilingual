"""Calibration speech.

The map fitted in :mod:`asr_align.interface` regresses one encoder's output onto
another's, so it needs audio both encoders can be run on -- and the target
encoder, VoiceChat's, only understands English. There is no such thing as a
multilingual target here: the thing being matched does not exist for Hindi. So
calibration is English, and whether the result carries to the other 39
language-locales is a property of the map, not of the data. That is the argument
for preferring the least expressive map that works; see the README.

LibriSpeech dev-clean is the default because it is 337 MB, public, already
16 kHz, and read speech at a level of clarity closer to someone talking to a
voice assistant than a conversational corpus would be.

Every clip is cropped to a fixed length so that a batch needs no padding and no
attention mask beyond the causal one. Cropping mid-utterance leaves the first
frames without left context, which is exactly what happens at the start of every
real turn, and it happens identically to both encoders.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Clip:
    path: Path
    offset: int
    n_samples: int

    @property
    def seconds(self) -> float:
        return self.n_samples / SAMPLE_RATE


def find_clips(
    root: Path,
    seconds: float = 6.0,
    limit: int | None = None,
    seed: int = 0,
    pattern: str = "**/*.flac",
) -> list[Clip]:
    """Fixed-length crops of every recording under ``root`` long enough for one.

    Sorted before shuffling so the same seed picks the same clips whatever order
    the filesystem hands them over in.
    """

    import soundfile

    want = int(round(seconds * SAMPLE_RATE))
    paths = sorted(root.glob(pattern))
    if not paths:
        raise SystemExit(f"no audio matching {pattern} under {root}")
    random.Random(seed).shuffle(paths)

    clips: list[Clip] = []
    for path in paths:
        info = soundfile.info(str(path))
        if info.samplerate != SAMPLE_RATE:
            raise SystemExit(
                f"{path}: {info.samplerate} Hz, but the featurizer is 16 kHz only"
            )
        if info.frames < want:
            continue
        # start a little way in, past the breath before the first word
        offset = min(info.frames - want, SAMPLE_RATE // 4)
        clips.append(Clip(path, offset, want))
        if limit is not None and len(clips) >= limit:
            break
    if not clips:
        raise SystemExit(f"no recording under {root} reaches {seconds:.1f} s")
    return clips


def load(clip: Clip) -> torch.Tensor:
    import soundfile

    samples, rate = soundfile.read(
        str(clip.path), start=clip.offset, frames=clip.n_samples, dtype="float32"
    )
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{clip.path}: {rate} Hz")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return torch.from_numpy(samples)


def batches(clips: list[Clip], batch_size: int) -> Iterator[torch.Tensor]:
    """`(batch, n_samples)` waveforms. Every clip is the same length by design."""

    for start in range(0, len(clips), batch_size):
        chunk = clips[start:start + batch_size]
        yield torch.stack([load(clip) for clip in chunk])


def split(clips: list[Clip], holdout: float = 0.2) -> tuple[list[Clip], list[Clip]]:
    """Fit and report on different speakers' worth of audio.

    LibriSpeech paths are `<speaker>/<chapter>/<utterance>.flac`, so splitting on
    the parent-of-parent directory keeps a speaker out of both halves. A map that
    only works on the speakers it was fitted on is not one to ship.
    """

    speakers = sorted({clip.path.parent.parent.name for clip in clips})
    n_held = max(1, int(round(len(speakers) * holdout)))
    held = set(speakers[:n_held])
    fit = [clip for clip in clips if clip.path.parent.parent.name not in held]
    evaluate = [clip for clip in clips if clip.path.parent.parent.name in held]
    if not fit or not evaluate:
        raise SystemExit(
            f"cannot hold out {holdout:.0%} of {len(speakers)} speakers; "
            "collect more calibration audio"
        )
    return fit, evaluate
