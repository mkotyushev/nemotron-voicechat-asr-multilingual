#!/usr/bin/env python3
"""
Fit the map that lets VoiceChat read a swapped-in ASR encoder, and write the
result as a checkpoint directory `convert_asr_to_mmproj.py` can build from on
its own.

What is wrong
-------------
`ASR_MODEL=multilingual` builds a perception mmproj whose encoder comes from
`nvidia/nemotron-3.5-asr-streaming-0.6b` and whose 1024 -> 4480 `proj` comes from
the VoiceChat container, because the ASR checkpoint has no equivalent. `proj` was
trained against the container's own encoder. The English encoder VoiceChat was
fine-tuned from is close enough to it that `proj` still reads it and the server
answers correctly; the multilingual one is not, and the server answers as if it
heard silence. See README, "Why the encoder cannot simply be swapped".

What this does
--------------
Runs both encoders on the same English speech, through the same featurizer the
deployment uses, and measures the drift the way Singh and Jaggi, "Model Fusion
via Optimal Transport" (NeurIPS 2020) does -- as a correspondence between two
sets of neurons, solved per width axis from their activations.

That produces two things:

  1. **Transport plans**, one per width group. These say whether the two
     encoders number their neurons the same way. Both are descendants of the
     same English checkpoint, so the expected answer is "yes, exactly", and
     `identity_agreement` in the report is where to check. A plan that is the
     identity is a real result: it rules out a permutation as the fix and sends
     the correction to the interface.

  2. **The interface map**, which is the artifact. After the last block there is
     no LayerNorm left, so any linear map on the encoder output folds into
     `proj` and changes nothing about the deployed graph. The report scores a
     ladder of them -- identity, the transport permutation, per-channel scale,
     Procrustes rotation and ridge -- on speakers held out of the fit, in the
     embedding space the language model actually reads.

     There is no rung above `linear`. Refitting `proj` from scratch, 1024 ->
     4480, looks more expressive and is not: ridge is linear in its target and
     the embedding target is `proj` applied to the hidden one, so the 4480-wide
     fit is exactly the 1024-wide fit composed with `proj`, for every penalty.
     A 1024 x 1024 correction is the most a shippable map can be.

Read the ladder before picking a rung. The regression target only exists for
English, because VoiceChat's encoder is the thing being matched and it
understands nothing else, so the most expressive map is also the one most able
to encode English-only structure. `--map auto` takes the best held-out R2;
naming a rung explicitly is the way to trade a little of that for a map more
likely to carry to the other 39 language-locales.

Usage
-----
    python align_asr.py \
        --asr-dir   /path/to/asr-multilingual \
        --container /path/to/nemotron_voicechat_11b-Q8_0.gguf \
        --audio     /path/to/LibriSpeech/dev-clean \
        -o          /path/to/asr-multilingual-aligned

What comes out is an ordinary checkpoint directory: the upstream encoder tensors
unchanged, plus the `proj` this fitted, plus the mel featurizer, a model card and
the fit report. Everything in this file -- torch, the calibration speech, the
transport solver -- is then build-time only. Serving a different encoder becomes
`--asr-dir <that directory>` and nothing else: no map to carry alongside it, and
no 12 GB container to read.

Point `--asr-dir` at `asr-en` to get the yardstick: that encoder is not the
container's either, and its `identity` row is how much mismatch the deployment
is known to tolerate and still answer "The capital of France is Paris."
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from asr_align import (
    data, encoder as encoder_module, export, features, hooks, interface, transport,
)
from asr_align.weights import load_asr, load_container, load_prompt_projector

logger = logging.getLogger("align-asr")

DEFAULT_WORK = Path(__file__).resolve().parent / ".cache" / "llama-voicechat.cpp"
# Open at both ends: a pick at either edge is reported, because it means the
# sweep did not contain the answer.
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--asr-dir", type=Path, required=True,
                    help="the encoder to align: config.json and model.safetensors")
    ap.add_argument("--container", type=Path, required=True,
                    help="nemotron_voicechat_11b-*.gguf: the target encoder, proj and the featurizer")
    ap.add_argument("--audio", type=Path, required=True,
                    help="directory of English calibration speech (LibriSpeech dev-clean)")
    ap.add_argument("--audio-glob", default="**/*.flac")
    ap.add_argument("-o", "--output", type=Path, required=True,
                    help="checkpoint directory to write")
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK,
                    help="llama-voicechat.cpp checkout, for gguf-py and vc_gguf")

    ap.add_argument("--clips", type=int, default=768, help="calibration clips (default 768)")
    ap.add_argument("--seconds", type=float, default=6.0, help="seconds per clip (default 6)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of speakers held out")
    ap.add_argument("--eval-frames", type=int, default=24000,
                    help="cap on held-out frames kept in memory for scoring")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--groups", default="residual,head",
                    help="transport groups to solve: residual, ffn, conv, head, or all")
    ap.add_argument("--ground-metric", default="correlation", choices=sorted(transport.GROUND_METRICS))
    ap.add_argument("--column-assignment", default="integral",
                    choices=sorted(transport.COLUMN_ASSIGNMENTS))

    ap.add_argument("--map", default="auto",
                    help="which rung to write: auto, identity, permutation, diagonal, "
                         "orthogonal, linear")
    ap.add_argument("--source-normalize", action="store_true",
                    help="feed the source encoder per-feature-normalized mel, which the "
                         "deployment does NOT do -- measures what enabling it would buy")
    ap.add_argument("--prompt-id", type=int, default=None,
                    help="also score the multilingual model's language head for this prompt id "
                         "(101 is 'auto'); report only, it needs a C++ change to ship")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    device = torch.device(args.device)

    # ------------------------------------------------------------- the models
    logger.info("target    : %s (stt_model.perception + proj)", args.container)
    target_weights = load_container(args.container, args.work)
    logger.info("source    : %s", args.asr_dir)
    source_weights = load_asr(args.asr_dir)

    mel_filters = target_weights["featurizer.fb"]
    window = target_weights["featurizer.window"]
    proj_weight = target_weights["proj.weight"].clone()
    proj_bias = target_weights["proj.bias"].clone()

    target = encoder_module.build(target_weights).to(device)
    source = encoder_module.build(source_weights).to(device)
    logger.info(
        "left context: source %d, target %d frames",
        source.hyper.attention_left_context,
        target.hyper.attention_left_context,
    )
    if source.hyper.n_embd != target.hyper.n_embd:
        raise SystemExit(
            f"widths differ: {source.hyper.n_embd} against {target.hyper.n_embd}. "
            "This aligns two encoders of one architecture, not two sizes."
        )

    prompt_head = None
    if args.prompt_id is not None:
        prompt_head = load_prompt_projector(args.asr_dir, args.prompt_id)
        if prompt_head is None:
            logger.warning("%s has no prompt_projector; --prompt-id ignored", args.asr_dir)
        else:
            prompt_head = {k: v.to(device) for k, v in prompt_head.items()}

    # --------------------------------------------------------------- the data
    clips = data.find_clips(
        args.audio, seconds=args.seconds, limit=args.clips, seed=args.seed,
        pattern=args.audio_glob,
    )
    fit_clips, eval_clips = data.split(clips, args.holdout)
    logger.info(
        "calibration: %d clips of %.1f s, %d fit / %d held out",
        len(clips), args.seconds, len(fit_clips), len(eval_clips),
    )

    # Featurized once and kept: the encoders are run over the fit set three
    # times (interface moments, stream statistics, head statistics) and the
    # float64 STFT is the slowest thing here. 768 six-second clips is ~240 MB.
    def featurize(subset: list[data.Clip]) -> list[torch.Tensor]:
        return [
            torch.stack([
                features.log_mel(wave, mel_filters, window).float() for wave in waveforms
            ])
            for waveforms in data.batches(subset, args.batch)
        ]

    def mels(cached: list[torch.Tensor]):
        for batch in cached:
            yield batch.to(device)

    fit_mels = featurize(fit_clips)
    eval_mels = featurize(eval_clips)

    # ---------------------------------------------------- interface moments
    n_embd = target.hyper.n_embd
    n_proj = proj_weight.shape[0]
    hidden_moments = interface.Moments(n_embd, n_embd)
    prompt_moments = interface.Moments(n_embd, n_embd) if prompt_head else None

    started = time.time()
    with torch.no_grad():
        for index, mel in enumerate(mels(fit_mels)):
            x = _source_hidden(source, mel, args.source_normalize)
            y = target(mel)
            embedding = target.project(y)
            flat_x = x.reshape(-1, n_embd).cpu()
            flat_y = y.reshape(-1, n_embd).cpu()
            hidden_moments.update(flat_x, flat_y)
            if prompt_moments is not None:
                prompt_moments.update(_prompt(prompt_head, x).reshape(-1, n_embd).cpu(), flat_y)
            if index % 10 == 0:
                logger.debug("moments: batch %d", index + 1)
    logger.info("fit frames: %d (%.0f s)", hidden_moments.count, time.time() - started)

    # ------------------------------------------------------ held-out frames
    eval_x, eval_e, eval_prompt = [], [], []
    kept = 0
    with torch.no_grad():
        for mel in mels(eval_mels):
            if kept >= args.eval_frames:
                break
            x = _source_hidden(source, mel, args.source_normalize)
            embedding = target.project(target(mel))
            eval_x.append(x.reshape(-1, n_embd).cpu())
            eval_e.append(embedding.reshape(-1, n_proj).cpu())
            if prompt_head is not None:
                eval_prompt.append(_prompt(prompt_head, x).reshape(-1, n_embd).cpu())
            kept += eval_x[-1].shape[0]
    eval_x = torch.cat(eval_x)[: args.eval_frames]
    eval_e = torch.cat(eval_e)[: args.eval_frames]
    eval_prompt = torch.cat(eval_prompt)[: args.eval_frames] if eval_prompt else None
    logger.info("held-out frames: %d", eval_x.shape[0])

    # ------------------------------------------------------- transport plans
    kinds = _groups(args.groups)
    plans: dict[str, transport.TransportOperators] = {}
    plan_report: list[dict] = []
    if kinds:
        logger.info("transport groups: %s", ", ".join(sorted(kinds)))
        statistics = hooks.collect(source, target, mels(fit_mels), kinds,
                                   progress=logger.debug)
        for group, stats in statistics.items():
            operators, summary = transport.solve_group(
                group, stats,
                ground_metric=args.ground_metric,
                column_assignment=args.column_assignment,
            )
            plans[group] = operators
            plan_report.append(summary)
        if "head" in kinds and hooks.RESIDUAL_GROUP in plans:
            head_stats = hooks.collect_heads(
                source, target, mels(fit_mels), plans[hooks.RESIDUAL_GROUP],
                progress=logger.debug,
            )
            for group, stats in head_stats.items():
                operators, summary = transport.solve_group(
                    group, stats,
                    ground_metric=args.ground_metric,
                    # a mixture of heads is not a head
                    column_assignment="integral",
                )
                plans[group] = operators
                plan_report.append(summary)
        elif "head" in kinds:
            logger.warning("head plans need the residual plan; add residual to --groups")
        for row in plan_report:
            logger.info(
                "  %-18s %4d -> %-4d  cost %.4f (identity %.4f)  same index %.2f",
                row["group"], row["source_size"], row["target_size"],
                row["transport_cost"], row["identity_cost"], row["identity_agreement"],
            )

    # ------------------------------------------------------------ the ladder
    projection = (proj_weight, proj_bias)
    candidates: list[interface.AffineMap] = [interface.identity_map(hidden_moments)]
    if hooks.RESIDUAL_GROUP in plans:
        candidates.append(
            interface.permutation_map(hidden_moments, plans[hooks.RESIDUAL_GROUP].permutation())
        )
    candidates.append(interface.diagonal_map(hidden_moments))
    candidates.append(interface.orthogonal_map(hidden_moments))

    rows: list[dict] = []
    for candidate in candidates:
        metrics = interface.score(candidate, eval_x, eval_e, projection=projection)
        deployed = interface.score_deployed(candidate, eval_x, eval_e, projection)
        rows.append({
            "map": candidate.name, **metrics, **candidate.detail,
            "deployed_r2": deployed["r2"], "deployed_cosine_mean": deployed["cosine_mean"],
        })

    linear, linear_trace = interface.select_alpha(
        hidden_moments, eval_x, eval_e, RIDGE_ALPHAS, projection=projection, name="linear"
    )
    candidates.append(linear)
    rows.append({
        "map": "linear",
        **interface.score(linear, eval_x, eval_e, projection=projection),
        **interface.score_deployed_row(linear, eval_x, eval_e, projection),
        **linear.detail, "sweep": linear_trace,
    })
    _warn_edge(linear_trace, linear.detail.get("alpha"), "linear")

    if prompt_moments is not None and eval_prompt is not None:
        head_map, head_trace = interface.select_alpha(
            prompt_moments, eval_prompt, eval_e, RIDGE_ALPHAS,
            projection=projection, name="prompt+linear",
        )
        rows.append({
            "map": "prompt+linear",
            "shippable": False,
            **interface.score(head_map, eval_prompt, eval_e, projection=projection),
            **interface.score_deployed_row(head_map, eval_prompt, eval_e, projection),
            **head_map.detail,
            "sweep": head_trace,
        })
        _warn_edge(head_trace, head_map.detail.get("alpha"), "prompt+linear")

    logger.info("held-out fit, in the 4480-wide embedding space:")
    for row in rows:
        logger.info(
            "  %-14s R2 %+.4f  cos %+.4f (p05 %+.4f)  | as deployed: R2 %+.4f  cos %+.4f%s",
            row["map"], row["r2"], row["cosine_mean"], row["cosine_p05"],
            row.get("deployed_r2", float("nan")),
            row.get("deployed_cosine_mean", float("nan")),
            "" if row.get("shippable", True) else "   [report only]",
        )

    # ----------------------------------------------------------- write it out
    shippable = {c.name: c for c in candidates}
    if args.map == "auto":
        chosen = max(shippable.values(), key=lambda c: _r2(rows, c.name))
        logger.info("--map auto picked %s", chosen.name)
    elif args.map in shippable:
        chosen = shippable[args.map]
    else:
        raise SystemExit(f"--map {args.map} is not one of: auto, {', '.join(shippable)}")

    new_weight, new_bias = chosen.fold_into_projection(proj_weight, proj_bias)
    metadata = {
        "map": chosen.name,
        "projection_dim": int(new_weight.shape[0]),
        "detail": {k: v for k, v in chosen.detail.items()},
        "source": str(args.asr_dir),
        "container": str(args.container),
        "audio": str(args.audio),
        "clips": len(clips),
        "seconds": args.seconds,
        "fit_frames": hidden_moments.count,
        "eval_frames": int(eval_x.shape[0]),
        "source_normalize": bool(args.source_normalize),
        "ground_metric": args.ground_metric,
        "column_assignment": args.column_assignment,
        "ladder": rows,
        "transport": plan_report,
    }
    # The encoder tensors go out as the published checkpoint has them, not as the
    # fit saw them: the F16 rounding `mmproj_precision` applies is what writing
    # the mmproj does anyway, and doing it here as well would round twice. A diff
    # against upstream should show only the tensors that were not there before.
    upstream = load_asr(args.asr_dir, mmproj_precision=False)
    export.export(
        args.output,
        source=args.asr_dir,
        encoder={k: v.numpy() for k, v in upstream.items() if k.startswith("encoder.")},
        proj_weight=new_weight.numpy(),
        proj_bias=new_bias.numpy(),
        featurizer={
            "fb": target_weights["featurizer.fb"].numpy(),
            "window": target_weights["featurizer.window"].numpy(),
        },
        report=metadata,
    )
    logger.info("wrote %s", args.output)
    logger.info("next: point the server's ASR_DIR at %s and rerun its convert.sh", args.output)


def _warn_edge(trace: list[dict], alpha: float | None, name: str) -> None:
    """A penalty chosen at the end of the sweep means the sweep was too short."""

    if alpha is None or not trace:
        return
    if alpha in (trace[0]["alpha"], trace[-1]["alpha"]):
        logger.warning(
            "%s picked alpha=%g, the %s of the sweep: widen RIDGE_ALPHAS or "
            "collect more calibration frames",
            name, alpha, "start" if alpha == trace[0]["alpha"] else "end",
        )


def _r2(rows: list[dict], name: str) -> float:
    """The score --map auto ranks on: the map as the server will run it."""

    for row in rows:
        if row["map"] == name and row.get("shippable", True):
            return float(row.get("deployed_r2", row["r2"]))
    return -float("inf")


def _source_hidden(model, mel: torch.Tensor, normalize: bool) -> torch.Tensor:
    """The source encoder's output, optionally on normalized features.

    The deployment does not normalize -- `mtmd.cpp` builds the parakeet
    preprocessor with `norm_per_feature = false` for this projector type -- but
    both published ASR checkpoints were trained with NeMo's per-feature
    normalization on, so the swapped-in encoder is being fed a distribution it
    has never seen. This flag measures what that costs. Only the source is
    normalized: the target is VoiceChat's own encoder, which is what it is.
    """

    if normalize:
        mel = torch.stack([features.per_feature_normalize(one) for one in mel])
    return model(mel)


def _prompt(head: dict[str, torch.Tensor], hidden: torch.Tensor) -> torch.Tensor:
    inner = torch.nn.functional.relu(
        hidden @ head["linear_1.weight"].t() + head["linear_1.bias"]
    )
    return inner @ head["linear_2.weight"].t() + head["linear_2.bias"]


def _groups(spec: str) -> set[str]:
    if spec in ("", "none"):
        return set()
    if spec == "all":
        return set(hooks.GROUP_KINDS)
    kinds = {part.strip() for part in spec.split(",") if part.strip()}
    unknown = kinds - set(hooks.GROUP_KINDS)
    if unknown:
        raise SystemExit(
            f"--groups {', '.join(sorted(unknown))} is not one of: "
            f"{', '.join(hooks.GROUP_KINDS)}, all, none"
        )
    return kinds


if __name__ == "__main__":
    main()
