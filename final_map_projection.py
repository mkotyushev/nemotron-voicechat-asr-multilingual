#!/usr/bin/env python3
"""Run Comparison 3: final activation-map projection only.

The encoder never moves.  What this runner fits is the correspondence between
the two pretrained encoders' final activations -- ``h_E A_L ~= h_M`` and
``h_M B_L ~= h_E``, both regularized toward the identity -- on LibriSpeech
map-training speakers, selects the regularization on held-out LibriSpeech
speakers, checks both maps on FLEURS without refitting, and folds only the
reverse map into the untouched FT_EN VoiceChat projection.  The exported
artifact carries ``PT_ML``'s encoder tensors byte for byte and differs from
Comparison 1 in ``proj.*`` alone, which is what makes the two comparable as a
measurement of interface alignment.

``PT_ML``'s side of the paired activations is read from the frozen Comparison 1
cache rather than recomputed, and one shard per split is re-encoded to prove the
cache still belongs to this checkpoint and this graph.  The evaluation passes,
the reference bundle, and the embedding writer are imported from the Comparison 2
runner so that rows from the two arms are produced by the same code, which is
what the shared result contract requires of them.

The reserved LibriSpeech test split is verified by the manifest consumer but is
never encoded or scored.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import direct_task_arithmetic as direct_runner
import pt_ml_baseline as baseline_runner
from asr_align import baseline, data, direct, encoder as encoder_module
from asr_align import evaluation, export, final_map, manifests
from asr_align.experiments import (
    ExperimentValidationError,
    assert_runtime_config_inherited,
    inherit_runtime_config,
    sha256_file,
    validate_encoder_triplet,
)
from asr_align.interface import Moments
from asr_align.weights import load_asr, load_mmproj, load_voicechat_safetensors

logger = logging.getLogger("final-map-projection")
REPOSITORY = Path(__file__).resolve().parent

_write_json = direct_runner._write_json
_file_record = direct_runner._file_record
_tree_records = direct_runner._tree_records
_as_numpy = direct_runner._as_numpy
_featurize_clips = direct_runner._featurize_clips
_read_audio = direct_runner._read_audio

# One re-encoded shard per split is enough to bind the frozen cache to this
# checkpoint; the shards are content-hashed on every read regardless.
CACHE_PARITY_SHARDS = 1
CACHE_PARITY_RELATIVE_L2 = 1e-6


def _clip_index(clips: Sequence[data.Clip], root: Path) -> dict[str, data.Clip]:
    index: dict[str, data.Clip] = {}
    for clip in clips:
        index[clip.path.resolve().relative_to(root).as_posix()] = clip
    return index


def _clips_for(records: Sequence[str], index: Mapping[str, data.Clip]) -> list[data.Clip]:
    missing = [record for record in records if record not in index]
    if missing:
        raise ExperimentValidationError(
            f"the frozen activation cache names recordings outside this split: {missing[:4]}"
        )
    return [index[record] for record in records]


@torch.inference_mode()
def _final_activations(
    model: torch.nn.Module,
    clips: Sequence[data.Clip],
    *,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    label: str,
) -> torch.Tensor:
    mel = _featurize_clips(clips, mel_filters, window).to(device)
    hidden = model(mel)
    if not bool(torch.isfinite(hidden).all()):
        raise ExperimentValidationError(f"{label} final-layer activations are not finite")
    return hidden


def _flatten(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(-1, value.shape[-1])


@torch.inference_mode()
def _fit_moments(
    pt_en_model: torch.nn.Module,
    pt_ml_model: torch.nn.Module,
    cache: final_map.ActivationCache,
    index: Mapping[str, data.Clip],
    *,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    width: int,
) -> tuple[Moments, Moments, dict[str, Any]]:
    """Accumulate both directions' cross-moments over LibriSpeech map_train."""

    forward = Moments(width, width)
    reverse = Moments(width, width)
    parity: list[dict[str, Any]] = []
    shards = cache.records("map_train")
    total = sum(len(shard["records"]) for shard in shards)
    clips_seen = 0
    for position, (shard, records, cached) in enumerate(
        final_map.iter_shards(cache, "map_train")
    ):
        clips = _clips_for(records, index)
        h_e = _final_activations(
            pt_en_model,
            clips,
            device=device,
            mel_filters=mel_filters,
            window=window,
            label="E/PT_EN map_train",
        )
        h_m = cached.to(device)
        if h_e.shape != h_m.shape:
            raise ExperimentValidationError(
                f"PT_EN produced {tuple(h_e.shape)} against the frozen PT_ML "
                f"{tuple(h_m.shape)}; broadcasting is forbidden"
            )
        if position < CACHE_PARITY_SHARDS:
            recomputed = _final_activations(
                pt_ml_model,
                clips,
                device=device,
                mel_filters=mel_filters,
                window=window,
                label="M/PT_ML map_train",
            )
            parity.append(
                _cache_parity_row(shard, "map_train", recomputed, h_m)
            )
            del recomputed
        forward.update(_flatten(h_e), _flatten(h_m))
        reverse.update(_flatten(h_m), _flatten(h_e))
        clips_seen += len(clips)
        if (position + 1) % 25 == 0 or position + 1 == len(shards):
            logger.info("map-train moments: %d/%d clips", clips_seen, total)
        del h_e, h_m
    summary = {
        "split": "LibriSpeech/map_train",
        "clips": clips_seen,
        "frames": int(forward.count),
        "shards": len(cache.records("map_train")),
        "cache_parity": parity,
    }
    return forward, reverse, summary


def _cache_parity_row(
    shard: Mapping[str, Any], split: str, recomputed: torch.Tensor, cached: torch.Tensor
) -> dict[str, Any]:
    change = baseline.numeric_change(cached, recomputed)
    if change["relative_l2"] > CACHE_PARITY_RELATIVE_L2:
        raise ExperimentValidationError(
            f"the frozen Comparison 1 {split} activations do not reproduce with this "
            f"PT_ML checkpoint: relative L2 {change['relative_l2']:g}"
        )
    return {
        "split": split,
        "shard": str(shard["path"]),
        "tolerance_relative_l2": CACHE_PARITY_RELATIVE_L2,
        "passed": True,
        **change,
    }


@torch.inference_mode()
def _held_out_activations(
    pt_en_model: torch.nn.Module,
    pt_ml_model: torch.nn.Module,
    cache: final_map.ActivationCache,
    index: Mapping[str, data.Clip],
    *,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Collect the held-out speakers' paired final activations for selection."""

    english: list[torch.Tensor] = []
    multilingual: list[torch.Tensor] = []
    parity: list[dict[str, Any]] = []
    clips_seen = 0
    for position, (shard, records, cached) in enumerate(
        final_map.iter_shards(cache, "validation")
    ):
        clips = _clips_for(records, index)
        h_e = _final_activations(
            pt_en_model,
            clips,
            device=device,
            mel_filters=mel_filters,
            window=window,
            label="E/PT_EN validation",
        )
        if h_e.shape != cached.shape:
            raise ExperimentValidationError(
                f"PT_EN produced {tuple(h_e.shape)} against the frozen PT_ML "
                f"{tuple(cached.shape)} on validation"
            )
        if position < CACHE_PARITY_SHARDS:
            recomputed = _final_activations(
                pt_ml_model,
                clips,
                device=device,
                mel_filters=mel_filters,
                window=window,
                label="M/PT_ML validation",
            )
            parity.append(_cache_parity_row(shard, "validation", recomputed, cached.to(device)))
            del recomputed
        english.append(_flatten(h_e).float().cpu())
        multilingual.append(_flatten(cached).float())
        clips_seen += len(clips)
        del h_e
    summary = {
        "split": "LibriSpeech/validation",
        "clips": clips_seen,
        "frames": int(sum(int(value.shape[0]) for value in english)),
        "shards": len(cache.records("validation")),
        "cache_parity": parity,
    }
    return torch.cat(english), torch.cat(multilingual), summary


@torch.inference_mode()
def _fleurs_generalization(
    pt_en_model: torch.nn.Module,
    pt_ml_model: torch.nn.Module,
    payload: Mapping[str, Any],
    maps: Mapping[str, Any],
    *,
    device: torch.device,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
) -> dict[str, Any]:
    """Score both already-selected maps on FLEURS. Nothing here is refitted.

    FLEURS is evaluation-only, so this runs after selection and cannot reach
    back into it.  The English takes are kept beside the foreign ones because
    they separate two different failures: a map that only works on LibriSpeech,
    and a map that only works on English.
    """

    root = Path(str(payload["root"])).resolve()
    groups = ("english_reference", "english_query", "foreign_query")
    rows: dict[str, Any] = {}
    for language in payload["languages"]:
        pairs = payload["pairs"][language]
        language_rows: dict[str, Any] = {}
        for group in groups:
            accumulators = {
                name: final_map.ScoreAccumulator(mapping)
                for name, mapping in maps.items()
            }
            paths = sorted({manifests.resolve_take(root, row[group]).resolve() for row in pairs})
            for path in paths:
                mel = _mel_for(_read_audio(path), mel_filters, window).to(device)
                h_e = _flatten(pt_en_model(mel))
                h_m = _flatten(pt_ml_model(mel))
                if h_e.shape != h_m.shape:
                    raise ExperimentValidationError(
                        f"{path}: PT_EN and PT_ML disagree on frame count "
                        f"({tuple(h_e.shape)} against {tuple(h_m.shape)})"
                    )
                for name, accumulator in accumulators.items():
                    source, target = (h_e, h_m) if name.startswith(final_map.FORWARD) else (h_m, h_e)
                    accumulator.update(source, target)
            language_rows[group] = {
                "language": language,
                "take": group,
                "recordings": len(paths),
                "maps": {name: accumulator.result() for name, accumulator in accumulators.items()},
            }
            logger.info(
                "FLEURS %s/%s: %d recordings scored without refitting", language, group, len(paths)
            )
        rows[language] = language_rows
    return {
        "policy": "evaluation only; no map was fitted, selected, or adjusted on FLEURS",
        "languages": rows,
    }


def _mel_for(samples: torch.Tensor, mel_filters: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    from asr_align import features

    return features.log_mel(samples, mel_filters, window).float()[None]


def _map_arrays(name: str, mapping: Any) -> dict[str, np.ndarray]:
    return {
        f"{name}.weight": mapping.weight.double().float().numpy(),
        f"{name}.bias": mapping.bias.double().float().numpy(),
    }


def _write_delta_markdown(path: Path, report: Mapping[str, Any]) -> None:
    english = report["english_voicechat_space"]
    lines = [
        f"# Comparison 3 against Comparison 1 ({report['precision']})",
        "",
        "The encoder is byte-identical to `PT_ML` in both rows, so every difference",
        "below is the effect of aligning the final activation interface alone.",
        "",
        "## English VoiceChat space",
        "",
        "| metric | PT_ML | final map | difference | paired 95% CI |",
        "|---|---:|---:|---:|:---:|",
    ]
    for metric, row in english.items():
        interval = row.get("paired_interval")
        bounds = (
            f"[{interval['low']:+.4f}, {interval['high']:+.4f}]" if interval else "—"
        )
        lines.append(
            f"| {metric} | {row['pt_ml']:+.6f} | {row['final_map']:+.6f} | "
            f"{row['difference']:+.6f} | {bounds} |"
        )
    for task, groups in report["retrieval"].items():
        lines += [
            "",
            f"## {task}",
            "",
            "| group | metric | PT_ML | final map | difference | paired 95% CI |",
            "|---|---|---:|---:|---:|:---:|",
        ]
        for group, row in sorted(groups.items()):
            for metric in evaluation.RETRIEVAL_METRICS:
                cell = row[metric]
                interval = cell["paired_interval"]
                lines.append(
                    f"| {group} | {metric} | {cell['pt_ml']:.6f} | {cell['final_map']:.6f} | "
                    f"{cell['difference']:+.6f} | "
                    f"[{interval['low']:+.4f}, {interval['high']:+.4f}] |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_map_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Comparison 3 final activation maps",
        "",
        "Fitted on LibriSpeech `map_train`, regularization selected on held-out",
        "LibriSpeech speakers by target-space R2, then scored on FLEURS without",
        "refitting. Only the reverse map reaches the deployed projection.",
        "",
        "| direction | formula | alpha | held-out R2 | held-out cosine | identity R2 | "
        "condition | identity distance |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for direction in (final_map.FORWARD, final_map.REVERSE):
        row = report["maps"][direction]
        lines.append(
            "| {direction} | `{formula}` | {alpha:g} | {r2:+.6f} | {cosine:.6f} | "
            "{identity:+.6f} | {condition:.1f} | {distance:.4f} |".format(
                direction=direction,
                formula=row["formula"],
                alpha=row["alpha"],
                r2=row["held_out"]["r2"],
                cosine=row["held_out"]["cosine_mean"],
                identity=row["held_out_identity"]["r2"],
                condition=row["conditioning"]["condition_number"],
                distance=row["identity_distance"]["relative_frobenius"],
            )
        )
    lines += [
        "",
        "## FLEURS generalization (no refitting)",
        "",
        "| language | take | map | R2 | cosine | frames |",
        "|---|---|---|---:|---:|---:|",
    ]
    for language, takes in sorted(report["fleurs_generalization"]["languages"].items()):
        for take, row in takes.items():
            for name, metrics in sorted(row["maps"].items()):
                lines.append(
                    f"| {language} | {take} | {name} | {metrics['r2']:+.6f} | "
                    f"{metrics['cosine_mean']:.6f} | {metrics['n_frames']} |"
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _release(*values: Any) -> None:
    del values
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(args: argparse.Namespace) -> Path:
    shared = baseline.load_shared_setup(args.shared_setup, verify_checkpoint_hashes=True)
    reference = direct.load_baseline_reference(args.baseline, shared)
    work = (
        args.work.resolve()
        if args.work is not None
        else Path(str(reference.run["runtime_reader"]["path"])).resolve()
    )
    if not (work / "gguf-py").is_dir() or not (work / "tools" / "voicechat").is_dir():
        raise ExperimentValidationError(f"{work} is not the prepared runtime reader")
    runtime_reader = baseline_runner._runtime_reader_provenance(work)

    output = args.output.resolve()
    if output.exists():
        raise ExperimentValidationError(
            f"refusing to replace Comparison 3 output {output}; choose a new experiment directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(shared.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(shared.seed)
    torch.use_deterministic_algorithms(True)

    checkpoints = shared.value["checkpoints"]
    pt_en_path = Path(str(checkpoints["E"]["path"])).resolve()
    logger.info("loading E/PT_EN without deployment rounding: %s", pt_en_path)
    pt_en = load_asr(pt_en_path, mmproj_precision=False)
    logger.info("loading M/PT_ML without deployment rounding: %s", shared.pt_ml_path)
    pt_ml = load_asr(shared.pt_ml_path, mmproj_precision=False)
    logger.info("loading original F/FT_EN safetensors: %s", shared.ft_en_path)
    ft_en = load_voicechat_safetensors(shared.ft_en_path)
    states = validate_encoder_triplet(pt_en, pt_ml, ft_en)
    assert_runtime_config_inherited(inherit_runtime_config(pt_ml.config), pt_ml.config)
    if pt_en.n_embd != pt_ml.n_embd:
        raise ExperimentValidationError("PT_EN and PT_ML disagree on the encoder width")
    width = pt_ml.n_embd

    libri_payload = manifests.load_manifest(shared.librispeech_manifest)
    fleurs_payload = manifests.load_manifest(shared.fleurs_manifest)
    manifests.verify_audio_files(fleurs_payload, root=Path(str(fleurs_payload["root"])))
    frozen_clips = data.from_frozen_manifest(shared.librispeech_manifest)
    map_train_clips = frozen_clips["map_train"]
    validation_clips = frozen_clips["validation"]
    if not map_train_clips or not validation_clips:
        raise ExperimentValidationError("the frozen LibriSpeech fitting splits are empty")
    libri_root = Path(str(libri_payload["root"])).resolve()
    eval_frames = int(reference.run["evaluation"]["english_frame_cap"])
    mel_filters = ft_en["featurizer.fb"]
    window = ft_en["featurizer.window"]

    cache = final_map.load_activation_cache(
        reference.root / str(reference.run["reference_cache"]["activations"]["path"]),
        manifest_sha256=shared.manifest_hashes["librispeech"],
        n_layer=pt_ml.n_layer,
        candidate_id=baseline.BASELINE_CANDIDATE_ID,
    )
    if sha256_file(cache.index_path) != reference.run["reference_cache"]["activations"]["sha256"]:
        raise ExperimentValidationError("the Comparison 1 activation index SHA-256 changed")

    stage_root = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        copied_setup = stage_root / "shared_setup.json"
        shutil.copyfile(shared.path, copied_setup)
        if sha256_file(copied_setup) != shared.sha256:
            raise ExperimentValidationError("copied shared_setup.json changed bytes")
        command = [str(Path(sys.executable).resolve()), *sys.argv]

        pt_en_model = encoder_module.build(pt_en).to(device)
        pt_ml_model = encoder_module.build(pt_ml).to(device)
        map_index = _clip_index(map_train_clips, libri_root)
        validation_index = _clip_index(validation_clips, libri_root)

        logger.info("collecting paired final-layer activations on LibriSpeech/map_train")
        forward_moments, reverse_moments, fit_summary = _fit_moments(
            pt_en_model,
            pt_ml_model,
            cache,
            map_index,
            device=device,
            mel_filters=mel_filters,
            window=window,
            width=width,
        )
        logger.info("collecting held-out paired activations on LibriSpeech/validation")
        held_english, held_multilingual, held_summary = _held_out_activations(
            pt_en_model,
            pt_ml_model,
            cache,
            validation_index,
            device=device,
            mel_filters=mel_filters,
            window=window,
        )

        logger.info("fitting identity-regularized ridge maps in both directions")
        forward_map, forward_trace = final_map.select_alpha(
            forward_moments, held_english, held_multilingual, name="A_L"
        )
        reverse_map, reverse_trace = final_map.select_alpha(
            reverse_moments, held_multilingual, held_english, name="B_L"
        )
        identity = final_map.identity_like(forward_map)
        map_records = {
            final_map.FORWARD: final_map.map_report(
                forward_map,
                direction=final_map.FORWARD,
                selection={
                    "split": "LibriSpeech/validation",
                    "criterion": "held-out target-space R2",
                    "alphas": list(final_map.ALPHAS),
                    "fleurs_used": False,
                    "trace": forward_trace,
                },
                held_out=final_map.score_map(forward_map, held_english, held_multilingual),
                fit=fit_summary,
            ),
            final_map.REVERSE: final_map.map_report(
                reverse_map,
                direction=final_map.REVERSE,
                selection={
                    "split": "LibriSpeech/validation",
                    "criterion": "held-out target-space R2",
                    "alphas": list(final_map.ALPHAS),
                    "fleurs_used": False,
                    "trace": reverse_trace,
                },
                held_out=final_map.score_map(reverse_map, held_multilingual, held_english),
                fit=fit_summary,
            ),
        }
        map_records[final_map.FORWARD]["held_out_identity"] = final_map.score_map(
            identity, held_english, held_multilingual
        )
        map_records[final_map.REVERSE]["held_out_identity"] = final_map.score_map(
            identity, held_multilingual, held_english
        )
        for record in map_records.values():
            record["held_out"]["split"] = held_summary["split"]
            record["held_out"]["clips"] = held_summary["clips"]
        cycle = final_map.cycle_consistency(forward_map, reverse_map)
        cycle["held_out"] = {
            "E_to_M_to_E": final_map.score_map(
                final_map.compose(forward_map, reverse_map, "A_L B_L"),
                held_english,
                held_english,
            ),
            "M_to_E_to_M": final_map.score_map(
                final_map.compose(reverse_map, forward_map, "B_L A_L"),
                held_multilingual,
                held_multilingual,
            ),
            "split": held_summary["split"],
        }

        logger.info("scoring both maps on FLEURS without refitting")
        fleurs_generalization = _fleurs_generalization(
            pt_en_model,
            pt_ml_model,
            fleurs_payload,
            {
                f"{final_map.FORWARD}/A_L": forward_map,
                f"{final_map.REVERSE}/B_L": reverse_map,
                f"{final_map.FORWARD}/identity": identity,
                f"{final_map.REVERSE}/identity": identity,
            },
            device=device,
            mel_filters=mel_filters,
            window=window,
        )
        del pt_en_model, pt_ml_model
        _release()

        folded_weight, folded_bias, fold_report = final_map.fold_reverse_map(
            reverse_map, ft_en["proj.weight"], ft_en["proj.bias"]
        )
        fold_report["verification"] = final_map.verify_folding(
            reverse_map,
            ft_en["proj.weight"],
            ft_en["proj.bias"],
            folded_weight,
            folded_bias,
            held_multilingual[: final_map.SCORE_CHUNK_FRAMES],
        )
        fold_report["change_vs_original_projection"] = {
            "weight": baseline.numeric_change(ft_en["proj.weight"], folded_weight),
            "bias": baseline.numeric_change(ft_en["proj.bias"], folded_bias),
        }
        map_analysis = {
            "schema_version": "1.0",
            "comparison": final_map.COMPARISON,
            "maps": map_records,
            "cycle_consistency": cycle,
            "fleurs_generalization": fleurs_generalization,
            "projection_fold": fold_report,
            "selection_policy": {
                "map_fit": "LibriSpeech/map_train",
                "regularization_selection": "LibriSpeech/validation",
                "fleurs_tuning_allowed": False,
            },
        }
        map_report_path = stage_root / "analysis" / "final_activation_maps.json"
        _write_json(map_report_path, map_analysis)
        _write_map_markdown(stage_root / "analysis" / "final_activation_maps.md", map_analysis)
        maps_path = stage_root / "maps" / "final_activation_maps.safetensors"
        export.write_safetensors(
            maps_path,
            {
                **_map_arrays("forward.A_L", forward_map),
                **_map_arrays("reverse.B_L", reverse_map),
                "projection.folded.weight": _as_numpy(folded_weight),
                "projection.folded.bias": _as_numpy(folded_bias),
            },
            {
                "comparison": str(final_map.COMPARISON),
                "candidate_id": final_map.CANDIDATE_ID,
                "forward_alpha": str(forward_map.detail["alpha"]),
                "reverse_alpha": str(reverse_map.detail["alpha"]),
                "librispeech_manifest_sha256": shared.manifest_hashes["librispeech"],
                "fit_split": "map_train",
                "selection_split": "validation",
            },
        )
        del held_english, held_multilingual, forward_moments, reverse_moments
        _release()

        name = final_map.CANDIDATE_ID
        attached = baseline.attach_voicechat_interface(pt_ml, ft_en)
        interface_equality = baseline.assert_exact_tensors(
            ft_en, attached, keys=("featurizer.fb", "featurizer.window")
        )
        original_projection = baseline.assert_exact_tensors(
            ft_en, attached, keys=("proj.weight", "proj.bias")
        )
        attached["proj.weight"] = folded_weight.contiguous()
        attached["proj.bias"] = folded_bias.contiguous()
        assert_runtime_config_inherited(attached.config, pt_ml.config)
        encoder_identity = final_map.assert_encoder_byte_identical(
            pt_ml, attached, label="candidate against PT_ML"
        )

        artifact = stage_root / "artifacts" / name
        artifact_report: dict[str, Any] = {
            "schema_version": "1.0",
            "artifact_kind": final_map.ARTIFACT_KIND,
            "comparison": final_map.COMPARISON,
            "candidate_id": name,
            "lambda": final_map.CANDIDATE_LAMBDA,
            "map": "identity-regularized ridge on the final activations (reverse direction)",
            "alpha": float(reverse_map.detail["alpha"]),
            "held_out_r2": float(map_records[final_map.REVERSE]["held_out"]["r2"]),
            "formula": {
                "forward": "h_E A_L ~= h_M",
                "reverse": "h_M B_L ~= h_E",
                "projection": "W_proj,M = W_proj,F B_L^T; b_proj,M = W_proj,F b_L + b_proj,F",
            },
            "projection_dim": int(attached["proj.weight"].shape[0]),
            "encoder_source": str(shared.pt_ml_path),
            "encoder_byte_identical_to_pt_ml": encoder_identity,
            "ft_en_interface_source": str(shared.ft_en_path),
            "shared_setup": str(shared.path),
            "shared_setup_sha256": shared.sha256,
            "manifests": shared.manifest_hashes,
            "runtime_configuration_source": "M/PT_ML",
            "runtime_configuration_exact": True,
            "maps": map_records,
            "projection_fold": fold_report,
            "command": command,
        }
        export.export(
            artifact,
            source=shared.pt_ml_path,
            encoder={
                key: _as_numpy(value)
                for key, value in attached.items()
                if key.startswith("encoder.")
            },
            proj_weight=_as_numpy(attached["proj.weight"]),
            proj_bias=_as_numpy(attached["proj.bias"]),
            featurizer={
                "fb": _as_numpy(attached["featurizer.fb"]),
                "window": _as_numpy(attached["featurizer.window"]),
            },
            report=artifact_report,
        )
        reloaded = load_asr(artifact, mmproj_precision=False)
        export_equality = baseline.assert_exact_tensors(attached, reloaded)
        reload_identity = final_map.assert_encoder_byte_identical(
            pt_ml, reloaded, label="exported artifact against PT_ML"
        )
        assert_runtime_config_inherited(reloaded.config, pt_ml.config)
        source_config = json.loads((shared.pt_ml_path / "config.json").read_text(encoding="utf-8"))
        artifact_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
        metadata = {
            "derived_from": artifact_config.pop("derived_from", None),
            "voicechat_final_activation_map": artifact_config.pop(
                "voicechat_final_activation_map", None
            ),
        }
        if artifact_config != source_config or metadata["voicechat_final_activation_map"] is None:
            raise ExperimentValidationError(
                "the export changed PT_ML config fields or omitted provenance"
            )

        baseline_artifact = load_asr(
            reference.root / str(reference.run["artifact"]["path"]), mmproj_precision=False
        )
        comparison_one_identity = final_map.assert_encoder_byte_identical(
            baseline_artifact, reloaded, label="Comparison 1 artifact"
        )
        baseline_model = encoder_module.build(baseline_artifact).to(device)
        first_mel = _featurize_clips(validation_clips[:1], mel_filters, window).to(device)
        baseline_sanity, baseline_profile = direct_runner._forward_check(baseline_model, first_mel)
        del baseline_model, baseline_artifact
        _release()

        model = encoder_module.build(reloaded).to(device)
        sanity, profile = direct_runner._forward_check(model, first_mel)
        growth = direct.activation_growth_report(baseline_profile, profile)

        pre_reference = direct_runner._read_reference_bundle(
            reference, "pre_quantization", fleurs_payload["languages"]
        )
        candidate_bundle = {
            "librispeech": direct_runner._collect_librispeech(
                model,
                validation_clips,
                batch_size=args.batch,
                eval_frames=eval_frames,
                device=device,
                mel_filters=mel_filters,
                window=window,
                candidate_name=name,
            ),
            "fleurs": direct_runner._collect_fleurs(
                model,
                fleurs_payload,
                device=device,
                mel_filters=mel_filters,
                window=window,
                candidate_name=name,
            ),
        }
        pre_result = direct_runner._evaluate_stage(
            candidate_name=name,
            weight=final_map.CANDIDATE_LAMBDA,
            stage="pre_quantization",
            candidate_bundle=candidate_bundle,
            reference_bundle=pre_reference,
            manifest_hashes=shared.manifest_hashes,
            seed=shared.seed,
            comparison=final_map.COMPARISON,
        )
        pre_result_path = stage_root / "results" / "pre_quantization.json"
        evaluation.write_result(pre_result_path, pre_result)
        pre_embeddings = stage_root / "embeddings" / "pre_quantization.safetensors"
        direct_runner._write_candidate_embeddings(
            pre_embeddings,
            candidate_bundle,
            candidate_name=name,
            weight=final_map.CANDIDATE_LAMBDA,
            stage="pre_quantization",
            manifest_hashes=shared.manifest_hashes,
            comparison=final_map.COMPARISON,
        )
        del candidate_bundle, model
        _release()

        deployment = stage_root / "deployment" / f"{name}-Q8_0.gguf"
        baseline_runner._convert_to_q8(artifact, deployment, work)
        actual_post = load_mmproj(deployment, work, config=reloaded.config)
        simulated_post = load_asr(artifact, mmproj_precision=True)
        simulation_equality = baseline.assert_exact_tensors(simulated_post, actual_post)
        quantization = baseline.quantization_report(reloaded, actual_post)
        post_model = encoder_module.build(actual_post).to(device)
        post_sanity, post_profile = direct_runner._forward_check(post_model, first_mel)
        post_growth = direct.activation_growth_report(baseline_profile, post_profile)
        post_bundle = {
            "librispeech": direct_runner._collect_librispeech(
                post_model,
                validation_clips,
                batch_size=args.batch,
                eval_frames=eval_frames,
                device=device,
                mel_filters=mel_filters,
                window=window,
                candidate_name=name + "-Q8_0",
            ),
            "fleurs": direct_runner._collect_fleurs(
                post_model,
                fleurs_payload,
                device=device,
                mel_filters=mel_filters,
                window=window,
                candidate_name=name + "-Q8_0",
            ),
        }
        post_reference = direct_runner._read_reference_bundle(
            reference, "post_quantization", fleurs_payload["languages"]
        )
        post_result = direct_runner._evaluate_stage(
            candidate_name=name,
            weight=final_map.CANDIDATE_LAMBDA,
            stage="post_quantization",
            candidate_bundle=post_bundle,
            reference_bundle=post_reference,
            manifest_hashes=shared.manifest_hashes,
            seed=shared.seed,
            comparison=final_map.COMPARISON,
        )
        evaluation.validate_precision_pair(pre_result, post_result)
        post_result_path = stage_root / "results" / "post_quantization.json"
        evaluation.write_result(post_result_path, post_result)
        post_embeddings = stage_root / "embeddings" / "post_quantization.safetensors"
        direct_runner._write_candidate_embeddings(
            post_embeddings,
            post_bundle,
            candidate_name=name,
            weight=final_map.CANDIDATE_LAMBDA,
            stage="post_quantization",
            manifest_hashes=shared.manifest_hashes,
            comparison=final_map.COMPARISON,
        )
        precision_delta = baseline.precision_metric_delta(pre_result, post_result)
        precision_delta_path = stage_root / "results" / "precision_delta.json"
        _write_json(precision_delta_path, precision_delta)
        parity = baseline_runner._runtime_parity(
            artifact,
            wav=args.parity_wav,
            runtime_log=args.runtime_log,
            device=device,
            work=work,
        )
        del post_bundle, post_reference, post_model, actual_post, simulated_post
        _release()

        delta_records: dict[str, Any] = {}
        for stage, candidate_result in (
            ("pre_quantization", pre_result),
            ("post_quantization", post_result),
        ):
            table = final_map.comparison_delta_table(candidate_result, reference.results[stage])
            table_json = stage_root / "analysis" / f"versus_comparison_1_{stage}.json"
            table_markdown = stage_root / "analysis" / f"versus_comparison_1_{stage}.md"
            _write_json(table_json, table)
            _write_delta_markdown(table_markdown, table)
            delta_records[stage] = {
                "json": _file_record(table_json, relative_to=stage_root),
                "table": _file_record(table_markdown, relative_to=stage_root),
            }

        artifact_report["checks"] = {
            "original_ft_en_featurizer": interface_equality,
            "original_ft_en_projection_before_folding": original_projection,
            "encoder_byte_identical": {
                "against_pt_ml_source": encoder_identity,
                "against_exported_artifact": reload_identity,
                "against_comparison_1_artifact": comparison_one_identity,
            },
            "export_reload": export_equality,
            "source_configuration_preserved": True,
            "added_configuration_metadata": metadata,
            "sanity": sanity,
            "comparison_1_sanity": baseline_sanity,
            "activation_growth": growth,
        }
        _write_json(artifact / "final_activation_map.json", artifact_report)

        run_report = {
            "schema_version": "1.0",
            "comparison": final_map.COMPARISON,
            "artifact_kind": final_map.ARTIFACT_KIND,
            "candidate_id": name,
            "status": "complete",
            "command": command,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "working_directory": str(REPOSITORY),
                "deterministic_algorithms": True,
                "seed": shared.seed,
            },
            "shared_setup": {
                "source": str(shared.path),
                "copied_path": "shared_setup.json",
                "sha256": shared.sha256,
                "manifests": shared.manifest_hashes,
            },
            "sources": {
                role: {
                    "repo_id": record["repo_id"],
                    "revision": record["revision"],
                    "files": record["files"],
                }
                for role, record in checkpoints.items()
            },
            "paired_pt_ml_reference": {
                "run": _file_record(reference.run_path),
                "embeddings": _file_record(reference.embeddings_path),
                "activations": _file_record(cache.index_path),
                "results": {
                    stage: _file_record(path) for stage, path in reference.result_paths.items()
                },
                "exact_frozen_arrays_used": True,
            },
            "runtime_reader": runtime_reader,
            "alignment": {
                "encoder_modified": False,
                "interface_modified": True,
                "formula": "W_proj,M = W_proj,F B_L^T",
                "alphas": list(final_map.ALPHAS),
                "selected_alpha": {
                    final_map.FORWARD: float(forward_map.detail["alpha"]),
                    final_map.REVERSE: float(reverse_map.detail["alpha"]),
                },
                "fit": fit_summary,
                "held_out": held_summary,
                "report": _file_record(map_report_path, relative_to=stage_root),
                "table": _file_record(
                    stage_root / "analysis" / "final_activation_maps.md", relative_to=stage_root
                ),
                "maps": _file_record(maps_path, relative_to=stage_root),
            },
            "artifact": {
                "path": artifact.relative_to(stage_root).as_posix(),
                "files": _tree_records(artifact),
            },
            "evaluation": {
                "split": "LibriSpeech/validation and frozen FLEURS",
                "english_frame_cap": eval_frames,
                "reserved_librispeech_test_encoded": False,
                "results": {
                    "pre_quantization": _file_record(pre_result_path, relative_to=stage_root),
                    "post_quantization": _file_record(post_result_path, relative_to=stage_root),
                    "precision_delta": _file_record(precision_delta_path, relative_to=stage_root),
                },
                "embeddings": {
                    "pre_quantization": _file_record(pre_embeddings, relative_to=stage_root),
                    "post_quantization": _file_record(post_embeddings, relative_to=stage_root),
                },
                "versus_comparison_1": delta_records,
            },
            "quantization": {
                "policy": shared.deployment_quantization,
                "artifact": _file_record(deployment, relative_to=stage_root),
                "actual_artifact_matches_rounding_model": simulation_equality,
                "weight_change": quantization,
                "forward_check": {"sanity": post_sanity, "activation_growth": post_growth},
                "runtime_parity": parity,
            },
        }
        run_path = stage_root / "run.json"
        _write_json(run_path, run_report)
        (stage_root / "run.json.sha256").write_text(
            sha256_file(run_path) + "  run.json\n", encoding="utf-8"
        )
        stage_root.rename(output)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    logger.info("Comparison 3 complete: %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shared-setup",
        type=Path,
        required=True,
        help="frozen shared_setup.json, or its containing directory",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="completed Comparison 1 output directory, or its run.json",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--work",
        type=Path,
        default=None,
        help="prepared runtime reader (defaults to the exact path recorded by Comparison 1)",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parity-wav", type=Path, default=None)
    parser.add_argument("--runtime-log", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.batch <= 0:
        raise SystemExit("--batch must be positive")
    try:
        run(args)
    except (ExperimentValidationError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Comparison 3 rejected: {error}") from error


if __name__ == "__main__":
    main()
