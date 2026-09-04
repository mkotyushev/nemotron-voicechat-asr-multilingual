#!/usr/bin/env python3
"""
Hold `asr_align.encoder` to the graph it claims to be.

Everything this package fits is a correction measured on activations, so the
PyTorch encoder has to be the encoder the server runs -- not a faithful
FastConformer, not the HF port, but `clip_graph_voicechat::build()` with the
featurizer `mtmd.cpp` pairs with it. A port that were subtly wrong would still
produce a plausible-looking alignment report and a mmproj that does not work,
and nothing else in the pipeline would notice.

The runtime already knows how to say what it computed. `MTMD_DEBUG_EMBEDDINGS`
makes clip.cpp print the shape of the projected embeddings, the first and last
sixteen values of frame 0, and the mean, standard deviation, minimum, maximum
and sum over all of them. This script computes the same numbers in PyTorch and
diffs them.

Getting the runtime side
------------------------
    docker compose exec -e MTMD_DEBUG_EMBEDDINGS=1 nemotron-voicechat \
        /app/llama-voicechat \
          -m      /models/nemotron_voicechat_11b-stt-llm-Q8_0.gguf \
          --mmproj /models/mmproj-voicechat-perception-Q8_0.gguf \
          --audio  /models/test_question.wav \
        2>&1 | tee runtime.log

Then:
    python check_encoder_parity.py \
        --container /path/to/nemotron_voicechat_11b-Q8_0.gguf \
        --wav       /path/to/test_question.wav \
        --runtime-log runtime.log

What "agrees" means
-------------------
Not bit-exactness, and not float32 rounding either. ggml's CUDA path for a Q8_0
weight quantizes the *activations* to eight bits as well and does the dot
product in integers, so every one of the twenty-four blocks introduces a few
tenths of a percent that this port, which multiplies in float32, does not. The
weights themselves are identical: `asr_align.weights` dequantizes the same Q8_0
blocks the runtime multiplies, and rounds to F16 exactly the tensors the mmproj
stores as F16.

So the yardstick is a fraction of the embedding's own standard deviation, not a
relative error per element. What would say the port is wrong is structural: a
different frame count, a sign flip, values that match a permutation of the
runtime's rather than the runtime's, or an error that grows steadily from frame 0
to the last frame.

Measured on `test_question_16k.wav`, worst element of frame 0:

    container encoder, its own proj          0.014 sigma
    safetensors encoder, the same proj       0.021 sigma
    safetensors encoder, a fitted proj       0.069 sigma

The third is higher for a reason worth knowing rather than hiding. A ridge fit
puts larger, more cancelling weights in `proj` than training does, so the eight
bits the runtime quantizes activations to cost more per element going through it.
It is per-element noise, not bias: on the same checkpoint, aggregate accuracy is
untouched -- `align_asr.py` scores every map a second time with the projection
quantized exactly as the converter writes it, and R2 moves by 0.0002. Hence
`--tolerance`, which defaults to the strict figure the trained projections meet.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from asr_align import encoder as encoder_module, features
from asr_align.weights import load_asr, load_container

DEFAULT_WORK = Path(__file__).resolve().parent / ".cache" / "llama-voicechat.cpp"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--container", type=Path, default=None,
                    help="the VoiceChat GGUF, for the reference encoder, proj and the "
                         "featurizer. Not needed when --asr-dir carries its own")
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--asr-dir", type=Path, default=None,
                    help="check a swapped-in encoder instead, against the same wav; "
                         "proj and the featurizer still come from the container")
    ap.add_argument("--runtime-log", type=Path, default=None,
                    help="output of a run with MTMD_DEBUG_EMBEDDINGS=1")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE,
                    help="allowed difference, in units of the embedding's standard "
                         f"deviation (default {TOLERANCE}; a fitted proj needs ~0.10)")
    args = ap.parse_args()

    weights = load_asr(args.asr_dir) if args.asr_dir else None
    if weights is not None and "proj.weight" in weights:
        # An aligned checkpoint is self-contained: it brings the featurizer and
        # the projection the runtime will load, so the container is not needed to
        # reproduce what the runtime computes.
        source = weights
    elif args.container is not None:
        source = load_container(args.container, args.work)
        weights = source if weights is None else weights
    else:
        raise SystemExit(
            f"{args.asr_dir} carries no proj or featurizer, so --container is needed"
        )
    mel_filters = source["featurizer.fb"]
    window = source["featurizer.window"]
    proj_weight = source["proj.weight"]
    proj_bias = source["proj.bias"]
    model = encoder_module.build(weights).to(args.device)

    samples = _read_wav(args.wav)
    mel = features.log_mel(samples, mel_filters, window).float()
    print(f"wav        : {args.wav} ({samples.shape[0] / features.SAMPLE_RATE:.2f} s)")
    print(f"mel        : {tuple(mel.shape)}")
    expected = features.frames_out(samples.shape[0])

    with torch.no_grad():
        hidden = model(mel[None].to(args.device))[0]
        embedding = hidden.to(proj_weight.device) @ proj_weight.t() + proj_bias
    embedding = embedding.double().cpu().numpy()

    print(f"encoder    : {tuple(hidden.shape)}, frames_out() says {expected}")
    print(f"embeddings : [{embedding.shape[1]}, {embedding.shape[0]}]")
    _print_block(embedding)

    if args.runtime_log is None:
        return
    runtime = _parse(args.runtime_log.read_text(errors="replace"))
    if runtime is None:
        raise SystemExit(f"no MTMD_DEBUG_EMBEDDINGS block in {args.runtime_log}")
    _compare(embedding, runtime, args.tolerance)


def _print_block(embedding: np.ndarray) -> None:
    first = embedding[0]
    print("Token 0 (first 16 values): " + " ".join(f"{v:.6f}" for v in first[:16]))
    print("Token 0 (last 16 values):  " + " ".join(f"{v:.6f}" for v in first[-16:]))
    print(
        f"Stats: mean={embedding.mean():.6f}, std={embedding.std():.6f}, "
        f"min={embedding.min():.6f}, max={embedding.max():.6f}, sum={embedding.sum():.6f}"
    )


# Everything is measured in units of the embedding's standard deviation, which
# is the only scale on this tensor that means anything: it is centred near zero,
# so a relative error on its mean is a ratio of two numbers that are both noise.
TOLERANCE = 0.05


def _compare(mine: np.ndarray, runtime: dict, tolerance: float = TOLERANCE) -> None:
    ok = True
    n_embd, n_tokens = runtime["shape"]
    if (n_tokens, n_embd) != mine.shape:
        print(f"MISMATCH shape: runtime [{n_embd}, {n_tokens}], torch {mine.shape[::-1]}")
        ok = False
    scale = max(float(mine.std()), 1e-9)
    print()
    print(f"all differences below are in units of the embedding sigma = {scale:.6f}")

    for label, offset, values in (
        ("first 16", 0, runtime["first"]),
        ("last 16", mine.shape[1] - 16, runtime["last"]),
    ):
        theirs = np.asarray(values, dtype=np.float64)
        ours = mine[0, offset:offset + theirs.shape[0]]
        worst = float(np.abs(ours - theirs).max()) / scale
        print(f"frame 0 {label:>8}: worst element {worst:.4f} sigma")
        ok &= worst < tolerance

    for key in ("mean", "std", "min", "max"):
        theirs = runtime["stats"][key]
        ours = float(getattr(mine, key)())
        delta = abs(ours - theirs) / scale
        print(f"{key:>16}: runtime {theirs:+.6f}  torch {ours:+.6f}  "
              f"({delta:.4f} sigma)")
        ok &= delta < tolerance

    print("PARITY OK" if ok else "PARITY FAILED")
    if not ok:
        raise SystemExit(1)


def _parse(text: str) -> dict | None:
    block = re.search(
        r"=== MTMD_DEBUG_EMBEDDINGS ===(.*?)=== END MTMD_DEBUG_EMBEDDINGS ===",
        text,
        re.S,
    )
    if block is None:
        return None
    body = block.group(1)
    shape = re.search(r"Shape:\s*\[(\d+),\s*(\d+)\]", body)
    first = re.search(r"first 16 values\):\s*([-\d\.eE\s]+)", body)
    last = re.search(r"last 16 values\):\s*([-\d\.eE\s]+)", body)
    stats = re.search(
        r"Stats:\s*mean=([-\d\.eE]+),\s*std=([-\d\.eE]+),\s*min=([-\d\.eE]+),\s*max=([-\d\.eE]+)",
        body,
    )
    if not (shape and first and last and stats):
        raise SystemExit("the MTMD_DEBUG_EMBEDDINGS block is there but does not parse")
    return {
        "shape": (int(shape.group(1)), int(shape.group(2))),
        "first": [float(v) for v in first.group(1).split()],
        "last": [float(v) for v in last.group(1).split()],
        "stats": {
            "mean": float(stats.group(1)),
            "std": float(stats.group(2)),
            "min": float(stats.group(3)),
            "max": float(stats.group(4)),
        },
    }


def _read_wav(path: Path) -> torch.Tensor:
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="float32")
    if rate != features.SAMPLE_RATE:
        raise SystemExit(f"{path}: {rate} Hz, but the featurizer is 16 kHz only")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return torch.from_numpy(samples)


if __name__ == "__main__":
    main()
