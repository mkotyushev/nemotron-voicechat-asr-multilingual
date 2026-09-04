#!/usr/bin/env python3
"""
Does the alignment carry to languages it was never fitted on?

`align_asr.py` can only fit against English, because the thing it matches --
VoiceChat's own encoder -- understands nothing else. Whether the resulting map
is a generic change of basis or an English-shaped one is therefore not something
the fit can answer, and reading nine free-form LLM replies to French audio is
not an answer either: with a chat model on the other end, a topical coincidence
is always available.

FLEURS is parallel. The same sentences are recorded in every language, so
"did the meaning survive" can be asked as retrieval instead of as vibes:

    embed 150 English recordings with the container encoder      -> the reference
    embed the same 150 sentences spoken in French                -> the probe
    rank every reference against every probe by cosine

If the probe encoder puts French speech where VoiceChat puts the English
sentence that means the same thing, the matching sentence ranks first. If it
does not, top-1 sits at chance, which is 1/150.

Four probes, which together say what is transfer and what is an artefact:

    en (second take)  the ceiling. Different speaker, same language, same
                      encoder -- this is how well pooled embeddings retrieve at
                      all, and nothing below can be expected to beat it.
    container         VoiceChat's English-only encoder on foreign speech. The
                      floor: whatever it scores is what leaks through acoustics
                      alone, with no multilingual model involved.
    multilingual      the swapped-in encoder as the deployment builds it today.
    multilingual+map  the same, with the alignment folded into proj.

Embeddings are mean-pooled over frames and centred on the dataset mean before
the cosine. Centring is not cosmetic: pooled speech embeddings share a large
common component, and without removing it every pair looks similar and the
ranking is decided by noise.

Usage
-----
    python crosslingual_probe.py \
        --fleurs    /path/to/fleurs \
        --container /path/to/nemotron_voicechat_11b-Q8_0.gguf \
        --asr-dir   /path/to/asr-multilingual-aligned \
        --languages fr_fr ru_ru de_de
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from asr_align import encoder as encoder_module, features
from asr_align.weights import load_asr, load_container

logger = logging.getLogger("crosslingual")

DEFAULT_WORK = Path(__file__).resolve().parent / ".cache" / "llama-voicechat.cpp"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fleurs", type=Path, required=True,
                    help="directory holding data/<lang>/dev.tsv and x/<lang>/dev/*.wav")
    ap.add_argument("--container", type=Path, required=True)
    ap.add_argument("--asr-dir", type=Path, required=True,
                    help="the aligned checkpoint align_asr.py wrote; its own proj is "
                         "the aligned probe, and the container's is the unaligned one")
    ap.add_argument("--languages", nargs="+", default=["fr_fr", "ru_ru", "de_de"])
    ap.add_argument("--reference", default="en_us")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-o", "--output", type=Path, default=None, help="write the table as json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(args.device)

    container_weights = load_container(args.container, args.work)
    mel_filters = container_weights["featurizer.fb"]
    window = container_weights["featurizer.window"]
    proj = (container_weights["proj.weight"].to(device), container_weights["proj.bias"].to(device))

    container = encoder_module.build(container_weights).to(device)
    swapped_weights = load_asr(args.asr_dir)
    if "proj.weight" not in swapped_weights:
        raise SystemExit(
            f"{args.asr_dir} carries no proj, so there is no aligned probe to run. "
            "Point --asr-dir at a directory align_asr.py wrote."
        )
    swapped = encoder_module.build(swapped_weights).to(device)
    aligned = (swapped.proj.weight, swapped.proj.bias)
    report = args.asr_dir / "alignment.json"
    if report.exists():
        logger.info("alignment : %s (map %s)", args.asr_dir,
                    json.loads(report.read_text()).get("map", "?"))

    takes = {lang: _takes(args.fleurs, lang) for lang in [args.reference] + args.languages}

    @torch.no_grad()
    def pooled(model, paths: list[Path]) -> torch.Tensor:
        """Mean encoder output per utterance, before any projection.

        `proj` is affine, so the mean of the projected frames is the projection
        of the mean frame. Pooling here rather than after means one forward pass
        serves every projection, which is what lets the aligned and unaligned
        multilingual probes share their encoder run.
        """

        out = []
        for path in paths:
            mel = features.log_mel(_read(path), mel_filters, window).float()[None].to(device)
            out.append(model(mel)[0].mean(dim=0))
        return torch.stack(out)

    def project(hidden: torch.Tensor, projection) -> np.ndarray:
        return (hidden @ projection[0].t() + projection[1]).double().cpu().numpy()

    table = []
    for lang in args.languages:
        shared = sorted(set(takes[args.reference]) & set(takes[lang]))
        if not shared:
            logger.warning("no shared sentences for %s", lang)
            continue
        logger.info("%s: %d sentences shared with %s", lang, len(shared), args.reference)

        foreign = [takes[lang][i][0] for i in shared]
        reference = project(pooled(container, [takes[args.reference][i][0] for i in shared]), proj)
        swapped_hidden = pooled(swapped, foreign)
        probes = {
            "en (second take)": project(
                pooled(container, [takes[args.reference][i][-1] for i in shared]), proj
            ),
            "container": project(pooled(container, foreign), proj),
            "multilingual": project(swapped_hidden, proj),
            "multilingual+map": project(swapped_hidden, aligned),
        }
        for name, probe in probes.items():
            top1, top5, median = _retrieval(probe, reference)
            row = {
                "language": lang, "probe": name, "n": len(shared),
                "top1": top1, "top5": top5, "median_rank": median,
                "chance_top1": 1.0 / len(shared),
            }
            table.append(row)
            logger.info(
                "  %-18s top1 %5.1f%%  top5 %5.1f%%  median rank %5.1f  (chance %4.1f%%)",
                name, 100 * top1, 100 * top5, median, 100 * row["chance_top1"],
            )

    if args.output:
        args.output.write_text(json.dumps(table, indent=2))
        logger.info("wrote %s", args.output)


def _takes(root: Path, lang: str) -> dict[str, list[Path]]:
    """Sentence id -> the recordings of it that are actually on disk."""

    out: dict[str, list[Path]] = defaultdict(list)
    tsv = root / "data" / lang / "dev.tsv"
    for row in csv.reader(tsv.open(encoding="utf-8"), delimiter="\t"):
        path = root / "x" / lang / "dev" / row[1]
        if path.exists():
            out[row[0]].append(path)
    return out


def _read(path: Path) -> torch.Tensor:
    import soundfile

    samples, rate = soundfile.read(str(path), dtype="float32")
    if rate != features.SAMPLE_RATE:
        raise SystemExit(f"{path}: {rate} Hz, but the featurizer is 16 kHz only")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return torch.from_numpy(samples)


def _retrieval(probe: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """Top-1, top-5 and median rank of the matching sentence.

    Both sides are centred on their own mean first. Pooled speech embeddings
    carry a large component that is the same for every utterance -- the encoder's
    idea of "this is speech" -- and leaving it in makes every cosine about 0.9
    and the ordering noise.
    """

    probe = probe - probe.mean(axis=0, keepdims=True)
    reference = reference - reference.mean(axis=0, keepdims=True)
    probe /= np.linalg.norm(probe, axis=1, keepdims=True).clip(1e-12)
    reference /= np.linalg.norm(reference, axis=1, keepdims=True).clip(1e-12)

    similarity = probe @ reference.T
    truth = similarity.diagonal()[:, None]
    ranks = (similarity > truth).sum(axis=1) + 1
    return float((ranks == 1).mean()), float((ranks <= 5).mean()), float(np.median(ranks))


if __name__ == "__main__":
    main()
