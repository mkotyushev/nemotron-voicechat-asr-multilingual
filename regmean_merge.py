#!/usr/bin/env python3
"""Run Comparison 6: RegMean++ merge with the original projection.

This is the training-free arm.  ``PT_ML`` and ``FT_EN`` are merged in closed
form -- one symmetric solve per dense linear, a simple average everywhere else
-- and the result is read through the *untouched* VoiceChat projection, so it
stays inside invariant 6 as written and nothing in the served graph is fitted.
``PT_EN`` takes no part in the merge: RegMean combines full weights rather than
deltas and needs no shared base.  It is still loaded and validated, because the
triplet check is what enforces the exact-key, exact-shape invariant.

What the merge is weighted by is the difference between the two candidates'
input second moments, so Dataset A is deliberately two corpora rather than one:
SLURP for ``FT_EN`` and Speech-MASSIVE fr/de/ru for ``PT_ML``, the same 18
domains and 60 intents in different languages.  Common audio would make the two
Gram matrices coincide and reduce Eq. 2 exactly to the unweighted mean.

The shrinkage ``alpha`` is selected on a held-out Dataset A split by agreement
at the *encoder output*, not by the per-layer regression residuals: RegMean++
solves each depth greedily and never looks downstream, and the encoder output is
the only quantity the frozen language model reads.  FLEURS is not consulted for
anything here, and neither is the speech-to-action pilot.

Alongside the primary candidate the runner builds the three reference points the
comparison needs to be interpretable -- plain RegMean, to isolate the
cross-layer correction; simple averaging, which is also arm ``E4``'s encoder for
Comparison 7; and a LayerNorm variant seeded from ``FT_EN``, because under ++ the
averaged non-linear tensors feed every downstream solve.  The evaluation passes
and the reference bundle are imported from the Comparison 2 runner so that every
arm's rows are produced by the same code.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import direct_task_arithmetic as direct_runner
import pt_ml_baseline as baseline_runner
from asr_align import baseline, data, direct, encoder as encoder_module
from asr_align import evaluation, export, manifests, regmean
from asr_align.experiments import (
    ExperimentValidationError,
    assert_runtime_config_inherited,
    inherit_runtime_config,
    sha256_file,
    validate_encoder_triplet,
)
from asr_align.weights import EncoderWeights, load_asr, load_mmproj, load_voicechat_safetensors

logger = logging.getLogger("regmean-merge")
REPOSITORY = Path(__file__).resolve().parent

_write_json = direct_runner._write_json
_file_record = direct_runner._file_record
_tree_records = direct_runner._tree_records
_as_numpy = direct_runner._as_numpy
_featurize_clips = direct_runner._featurize_clips

PRIMARY = "regmean-plus-plus"
#: Every arm the comparison builds, and the one sentence that says what each is
#: for.  Only the primary reaches a deployment artifact and a Q8 stage.
ARMS = {
    PRIMARY: "RegMean++ at the selected shrinkage; the comparison's candidate",
    "regmean-plain": "plain RegMean, isolating the cross-layer correction",
    "simple-average": "the unweighted mean, and arm E4's encoder for Comparison 7",
    "regmean-layernorm-ft-en": "RegMean++ with the LayerNorms seeded from FT_EN",
    "regmean-ood-gram": "RegMean++ with G_M re-collected off-domain",
}


def _plans(alpha: float) -> dict[str, regmean.MergePlan | None]:
    return {
        PRIMARY: regmean.MergePlan(alpha=alpha),
        "regmean-plain": regmean.MergePlan(alpha=alpha, cross_layer=False),
        "simple-average": None,
        "regmean-layernorm-ft-en": regmean.MergePlan(alpha=alpha, layernorm_source="F"),
        "regmean-ood-gram": regmean.MergePlan(alpha=alpha),
    }


def _mel_batches(
    clips: Sequence[data.Clip],
    mel_filters: torch.Tensor,
    window: torch.Tensor,
    *,
    batch_size: int,
) -> list[torch.Tensor]:
    return [
        _featurize_clips(clips[start:start + batch_size], mel_filters, window)
        for start in range(0, len(clips), batch_size)
    ]


def _dataset_a_clips(
    slurp: Path, speech_massive: Path
) -> tuple[dict[str, dict[str, list[data.Clip]]], dict[str, Any]]:
    """The two Gram corpora, verified, with ``F`` on SLURP and ``M`` on the rest."""

    payloads = {"F": manifests.load_manifest(slurp), "M": manifests.load_manifest(speech_massive)}
    for role, payload in payloads.items():
        manifests.validate_dataset_a_manifest(payload)
        if payload["candidate"].split("/")[0] != role:
            raise ExperimentValidationError(
                f"{payload['dataset']} is declared for {payload['candidate']}, not {role}"
            )
    clips = {
        role: {
            split: data.from_dataset_a_manifest(
                slurp if role == "F" else speech_massive, split=split
            )
            for split in manifests.DATASET_A_SPLITS
        }
        for role in payloads
    }
    for payload in payloads.values():
        manifests.assert_merge_selection_source(payload["dataset"], "heldout")
    for split in manifests.DATASET_A_SPLITS:
        # "Equalise total frames across candidates" is exact here because both
        # corpora are the same number of identical-length crops, so neither the
        # clip-length distribution nor the collection volume can act as a merge
        # coefficient.
        counts = {role: len(clips[role][split]) for role in clips}
        if len(set(counts.values())) != 1:
            raise ExperimentValidationError(
                f"Dataset A {split} is not equalized across candidates: {counts}"
            )
    provenance = {
        role: {
            "dataset": payload["dataset"],
            "manifest_sha256": payload["manifest_sha256"],
            "root": payload["root"],
            "crop_seconds": payload["crop_seconds"],
            "source": payload["source"],
            "selection": payload["selection"],
            "clips": {split: len(clips[role][split]) for split in manifests.DATASET_A_SPLITS},
        }
        for role, payload in payloads.items()
    }
    return clips, provenance


def _agreements(
    merged: Mapping[str, torch.Tensor],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    config: Mapping[str, Any],
    heldout: Mapping[str, list[torch.Tensor]],
    *,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Held-out output agreement against each candidate, on that candidate's own audio."""

    return {
        role: regmean.output_agreement(
            merged, states[role], config, heldout[role], device=device
        )
        for role in regmean.CANDIDATES
    }


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_alpha_markdown(path: Path, report: Mapping[str, Any]) -> None:
    header = (
        "| alpha | R2 vs PT_ML (fr/de/ru) | R2 vs FT_EN (en) | cosine vs PT_ML | "
        "cosine vs FT_EN | mean R2 | weight L2 / PT_ML | thinnest solve | selected |"
    )
    lines = [
        "# Comparison 6 shrinkage selection",
        "",
        "Held-out Dataset A, encoder output. FLEURS and the pilot are not consulted.",
        "",
        "`thinnest solve` is the smallest fraction of a layer's input directions the",
        "Gram actually determines; the rest stay at the candidates' mean.",
        "",
        header,
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["rows"]:
        if "failed" in row:
            lines.append(f"| {row['alpha']:g} | diverged | | | | | | | no |")
            continue
        lines.append(
            "| {alpha:g} | {m_r2:+.6f} | {f_r2:+.6f} | {m_cos:.6f} | {f_cos:.6f} | "
            "{score:+.6f} | {ratio:.3f} | {rank:.4f} | {selected} |".format(
                alpha=row["alpha"],
                m_r2=row["agreement"]["M"]["r2"],
                f_r2=row["agreement"]["F"]["r2"],
                m_cos=row["agreement"]["M"]["cosine_mean"],
                f_cos=row["agreement"]["F"]["cosine_mean"],
                score=row["selection_score"],
                ratio=row["weight_l2_ratio_vs_pt_ml"],
                rank=row["worst_effective_rank_fraction"],
                selected="yes" if row["alpha"] == report["selected_alpha"] else "no",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_arm_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Comparison 6 arms",
        "",
        f"All at alpha = {report['alpha']:g}, held-out Dataset A, encoder output.",
        "",
        "| arm | what it isolates | R2 vs PT_ML | R2 vs FT_EN | mean R2 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            "| `{arm}` | {note} | {m:+.6f} | {f:+.6f} | {score:+.6f} |".format(
                arm=row["arm"],
                note=ARMS[row["arm"]],
                m=row["agreement"]["M"]["r2"],
                f=row["agreement"]["F"]["r2"],
                score=row["selection_score"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_delta_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Comparison 6 against comparisons 1 and 2",
        "",
        f"Precision stage: `{report['precision']}`. Every difference is against "
        "Comparison 1, which is the only direction with a paired interval.",
        "",
        "## English VoiceChat space",
        "",
        "| metric | PT_ML | merge | merge - PT_ML | 95% CI | lambda=1 | lambda=1 - PT_ML |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for metric, row in report["english_voicechat_space"].items():
        other = row.get("comparison_2_lambda_1")
        interval = row["merge"]["paired_interval"]
        lines.append(
            "| {metric} | {pt:+.6f} | {merge:+.6f} | {diff:+.6f} | [{low:+.4f}, {high:+.4f}] | "
            "{other:+.6f} | {other_diff:+.6f} |".format(
                metric=metric,
                pt=row["pt_ml"],
                merge=row["merge"]["value"],
                diff=row["merge"]["difference_vs_pt_ml"],
                low=interval["low"],
                high=interval["high"],
                other=other["value"] if other else float("nan"),
                other_diff=other["difference_vs_pt_ml"] if other else float("nan"),
            )
        )
    for task, groups in report["retrieval"].items():
        lines += ["", f"## {task}", "",
                  "| group | metric | PT_ML | merge | merge - PT_ML | 95% CI | lambda=1 - PT_ML |",
                  "|---|---|---:|---:|---:|---|---:|"]
        for group, metrics in groups.items():
            for metric in ("top1", "top5", "mrr"):
                row = metrics[metric]
                other = row.get("comparison_2_lambda_1")
                interval = row["merge"]["paired_interval"]
                lines.append(
                    "| {group} | {metric} | {pt:.6f} | {merge:.6f} | {diff:+.6f} | "
                    "[{low:+.4f}, {high:+.4f}] | {other_diff:+.6f} |".format(
                        group=group, metric=metric, pt=row["pt_ml"],
                        merge=row["merge"]["value"],
                        diff=row["merge"]["difference_vs_pt_ml"],
                        low=interval["low"], high=interval["high"],
                        other_diff=other["difference_vs_pt_ml"] if other else float("nan"),
                    )
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_embeddings(
    path: Path,
    bundle: Mapping[str, Any],
    *,
    arm: str,
    stage: str,
    plan: Mapping[str, Any] | None,
    manifest_hashes: Mapping[str, str],
) -> None:
    arrays = {
        f"librispeech.validation.{name}": value
        for name, value in bundle["librispeech"].items()
    }
    for language, values in bundle["fleurs"].items():
        arrays.update({f"fleurs.{language}.{name}": value for name, value in values.items()})
    export.write_safetensors(
        path,
        arrays,
        {
            "comparison": str(regmean.COMPARISON),
            "candidate_id": arm,
            "merge": json.dumps(plan, sort_keys=True) if plan else "simple average",
            "precision": stage,
            "librispeech_manifest_sha256": manifest_hashes["librispeech"],
            "fleurs_manifest_sha256": manifest_hashes["fleurs"],
        },
    )


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
            f"refusing to replace Comparison 6 output {output}; choose a new experiment directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(shared.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(shared.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    checkpoints = shared.value["checkpoints"]
    pt_en_path = Path(str(checkpoints["E"]["path"])).resolve()
    logger.info("loading E/PT_EN without deployment rounding: %s", pt_en_path)
    pt_en = load_asr(pt_en_path, mmproj_precision=False)
    logger.info("loading M/PT_ML without deployment rounding: %s", shared.pt_ml_path)
    pt_ml = load_asr(shared.pt_ml_path, mmproj_precision=False)
    logger.info("loading original F/FT_EN safetensors: %s", shared.ft_en_path)
    ft_en = load_voicechat_safetensors(shared.ft_en_path)
    triplet = validate_encoder_triplet(pt_en, pt_ml, ft_en)
    states = {"M": triplet["M"], "F": triplet["F"]}
    # PT_EN is the invariant check and nothing else: RegMean merges full
    # weights, not deltas, so it has no shared base to subtract.
    del pt_en, triplet
    _release()
    config = inherit_runtime_config(pt_ml.config)
    assert_runtime_config_inherited(config, pt_ml.config)
    pt_ml_norm = direct.state_norm_report(states["M"])

    clips, dataset_a = _dataset_a_clips(args.slurp, args.speech_massive)
    mel_filters = ft_en["featurizer.fb"]
    window = ft_en["featurizer.window"]
    logger.info("featurizing Dataset A")
    gram_mels = {
        role: _mel_batches(clips[role]["gram"], mel_filters, window, batch_size=args.gram_batch)
        for role in regmean.CANDIDATES
    }
    heldout_mels = {
        role: _mel_batches(clips[role]["heldout"], mel_filters, window, batch_size=args.gram_batch)
        for role in regmean.CANDIDATES
    }
    ood_mels: dict[str, list[torch.Tensor]] | None = None
    ood_provenance: dict[str, Any] | None = None
    if args.ood_speech_massive is not None:
        ood_clips, ood_dataset = _dataset_a_clips(args.slurp, args.ood_speech_massive)
        ood_mels = {
            "F": gram_mels["F"],
            "M": _mel_batches(
                ood_clips["M"]["gram"], mel_filters, window, batch_size=args.gram_batch
            ),
        }
        ood_provenance = ood_dataset["M"]

    fleurs_payload = manifests.load_manifest(shared.fleurs_manifest)
    manifests.verify_audio_files(fleurs_payload, root=Path(str(fleurs_payload["root"])))
    validation_clips = data.from_frozen_manifest(shared.librispeech_manifest)["validation"]
    eval_frames = int(reference.run["evaluation"]["english_frame_cap"])
    first_mel = _featurize_clips(validation_clips[:1], mel_filters, window).to(device)
    pre_reference = direct_runner._read_reference_bundle(
        reference, "pre_quantization", fleurs_payload["languages"]
    )
    task_arithmetic: dict[str, dict[str, Any]] = {}
    if args.task_arithmetic is not None:
        for stage in evaluation.PRECISION_STAGES:
            path = args.task_arithmetic / "results" / direct.candidate_id(1.0) / f"{stage}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            evaluation.validate_result(value)
            if value.get("manifests") != shared.manifest_hashes:
                raise ExperimentValidationError(
                    "the Comparison 2 endpoint used different frozen manifests"
                )
            task_arithmetic[stage] = value

    command = [str(Path(sys.executable).resolve()), *sys.argv]
    stage_root = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        copied_setup = stage_root / "shared_setup.json"
        shutil.copyfile(shared.path, copied_setup)
        if sha256_file(copied_setup) != shared.sha256:
            raise ExperimentValidationError("copied shared_setup.json changed bytes")

        # ---- the PT_ML forward profile every arm's growth check is read against
        pt_ml_artifact = load_asr(
            reference.root / reference.run["artifact"]["path"], mmproj_precision=False
        )
        pt_ml_model = encoder_module.build(pt_ml_artifact).to(device)
        _, pt_ml_profile = direct_runner._forward_check(pt_ml_model, first_mel)
        del pt_ml_model, pt_ml_artifact
        _release()

        # ---- alpha grid, on held-out Dataset A only -----------------------
        manifests.assert_merge_selection_source(dataset_a["M"]["dataset"], "heldout")
        # A follow-up run that only adds an ablation arm inherits the shrinkage
        # instead of reselecting it, so the ablation is compared at the same
        # setting and the selection is made exactly once.
        grid = regmean.ALPHAS if args.alpha is None else (args.alpha,)
        alpha_rows: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        for alpha in grid:
            plan = regmean.MergePlan(alpha=alpha)
            logger.info("merging at alpha=%g", alpha)
            try:
                merged, report = regmean.merge_encoders(
                    states, config, gram_mels, plan, device=device,
                    progress=lambda message: logger.debug("alpha=%g %s", alpha, message),
                )
                agreement = _agreements(merged, states, config, heldout_mels, device=device)
            except regmean.MergeValidationError as error:
                # Eq. 2 is a least-squares solve against a Gram that speech
                # makes very ill-conditioned, and the shrinkage is what keeps it
                # in hand.  A grid point that diverges is a measurement, not a
                # reason to lose the other five.
                logger.warning("alpha=%g failed: %s", alpha, error)
                alpha_rows.append({"alpha": alpha, "failed": str(error)})
                _release()
                continue
            score = regmean.selection_score(agreement)
            norm = direct.state_norm_report(merged)
            logger.info(
                "alpha=%g held-out R2 M=%+.4f F=%+.4f mean=%+.4f, weight L2 %.3fx PT_ML",
                alpha, agreement["M"]["r2"], agreement["F"]["r2"], score,
                float(norm["l2"]) / max(float(pt_ml_norm["l2"]), 1e-30),
            )
            alpha_rows.append(
                {"alpha": alpha, "agreement": agreement, "selection_score": score,
                 "weight_l2_ratio_vs_pt_ml": float(norm["l2"]) / max(float(pt_ml_norm["l2"]), 1e-30),
                 "worst_condition_number": max(
                     (value["condition_number"] for value in report["layers"].values()
                      if value["condition_number"] is not None),
                     default=None,
                 ),
                 "worst_effective_rank_fraction": min(
                     (value["effective_rank"] / value["input_dimension"]
                      for value in report["layers"].values()),
                     default=None,
                 ),
                 "degenerate_solves": report["degenerate_solves"]}
            )
            if best is None or score > best["selection_score"]:
                best = {"alpha": alpha, "selection_score": score, "state": merged,
                        "report": report, "agreement": agreement}
            else:
                del merged, report
            _release()
        if best is None:
            raise ExperimentValidationError("every alpha in the frozen grid failed")
        alpha_report = {
            "schema_version": "1.0",
            "comparison": regmean.COMPARISON,
            "grid": list(grid),
            "full_grid": list(regmean.ALPHAS),
            "selection_performed": args.alpha is None,
            "inherited_from": args.alpha_source,
            "excluded": [1.0],
            "excluded_reason": "alpha=1 removes the shrinkage; the paper's Table 9 zeroes out there",
            "criterion": (
                "mean held-out encoder-output R2 against each candidate on that "
                "candidate's own Dataset A audio"
            ),
            "criterion_rationale": (
                "RegMean++ solves each depth greedily and does not control error at "
                "the encoder output, which is the only quantity the frozen language "
                "model reads; the per-layer residuals are therefore not the criterion"
            ),
            "selection_split": "Dataset A heldout",
            "fleurs_consulted": False,
            "pilot_consulted": False,
            "rows": alpha_rows,
            "selected_alpha": best["alpha"],
        }
        _write_json(stage_root / "analysis" / "alpha_grid.json", alpha_report)
        _write_alpha_markdown(stage_root / "analysis" / "alpha_grid.md", alpha_report)

        # ---- the arms ------------------------------------------------------
        alpha = best["alpha"]
        arm_states = {PRIMARY: best["state"]}
        arm_reports = {PRIMARY: best["report"]}
        arm_agreements = {PRIMARY: best["agreement"]}
        arm_plans: dict[str, Mapping[str, Any] | None] = {PRIMARY: best["report"]["routing"]["plan"]}
        for name, plan in _plans(alpha).items():
            if name == PRIMARY or (args.arm and name not in args.arm):
                continue
            if name == "regmean-ood-gram" and ood_mels is None:
                continue
            logger.info("building arm %s", name)
            if plan is None:
                merged = regmean.simple_average(states)
                report = {
                    "routing": {"plan": {"method": "simple average", "alpha": None,
                                         "non_linear_tensors": "simple average",
                                         "merged_depths": "all", "merged_module_kinds": ["all"],
                                         "layernorm_source": "average",
                                         "cross_layer_input": "not applicable"},
                                "counts": {"regmean": 0, "average": len(merged)}},
                    "gram": {"rows_per_candidate": None,
                             "note": "no Gram matrices; this arm is the unweighted mean"},
                    "layers": {},
                    "degenerate_solves": [],
                }
            else:
                merged, report = regmean.merge_encoders(
                    states, config, ood_mels if name == "regmean-ood-gram" else gram_mels,
                    plan, device=device,
                    progress=lambda message, name=name: logger.debug("%s %s", name, message),
                )
                if name == "regmean-ood-gram":
                    report["gram"]["multilingual_source"] = ood_provenance
            arm_states[name] = merged
            arm_reports[name] = report
            arm_agreements[name] = _agreements(
                merged, states, config, heldout_mels, device=device
            )
            arm_plans[name] = report["routing"]["plan"]
            _release()
        arm_report = {
            "schema_version": "1.0",
            "comparison": regmean.COMPARISON,
            "alpha": alpha,
            "declared_choices": {
                "merged_depths": "all, including the subsampling projection",
                "merged_module_kinds": list(regmean.MODULE_KINDS),
                "paper_finding_not_adopted_as_default": (
                    "the paper reports middle and deep layers preserving >98% of the "
                    "all-layer result and MLP linears outperforming attention linears; "
                    "the depth range and module subset are recorded here as choices "
                    "rather than inherited"
                ),
            },
            "rows": [
                {"arm": name, "purpose": ARMS[name], "plan": arm_plans[name],
                 "agreement": arm_agreements[name],
                 "selection_score": regmean.selection_score(arm_agreements[name]),
                 "degenerate_solves": arm_reports[name]["degenerate_solves"]}
                for name in arm_states
            ],
        }
        _write_json(stage_root / "analysis" / "arms.json", arm_report)
        _write_arm_markdown(stage_root / "analysis" / "arms.md", arm_report)
        for name, report in arm_reports.items():
            _write_json(stage_root / "analysis" / "merges" / f"{name}.json", report)

        # ---- artifacts and the shared evaluation ---------------------------
        arm_records: dict[str, Any] = {}
        pre_results: dict[str, dict[str, Any]] = {}
        primary_artifact: Path | None = None
        for name in arm_states:
            encoder_state = arm_states[name]
            weight_norm = direct.state_norm_report(encoder_state)
            weight_norm["l2_ratio_vs_pt_ml"] = float(weight_norm["l2"]) / max(
                float(pt_ml_norm["l2"]), 1e-30
            )
            if weight_norm["l2_ratio_vs_pt_ml"] > direct.MAX_ACTIVATION_GROWTH:
                raise ExperimentValidationError(
                    f"{name} encoder L2 growth exceeds the safety tripwire"
                )
            candidate_weights = EncoderWeights(
                encoder_state, inherit_runtime_config(pt_ml.config), name
            )
            attached = baseline.attach_voicechat_interface(candidate_weights, ft_en)
            assert_runtime_config_inherited(attached.config, pt_ml.config)
            interface_equality = baseline.assert_exact_tensors(
                ft_en, attached, keys=baseline.ATTACHED_KEYS
            )
            artifact = stage_root / "artifacts" / name
            artifact_report: dict[str, Any] = {
                "schema_version": "1.0",
                "artifact_kind": regmean.ARTIFACT_KIND,
                "comparison": regmean.COMPARISON,
                "candidate_id": name,
                "purpose": ARMS[name],
                "method": arm_plans[name]["method"],
                "alpha": arm_plans[name]["alpha"],
                "map": None,
                "lambda": None,
                "projection_dim": int(attached["proj.weight"].shape[0]),
                "source": str(shared.pt_ml_path),
                "ft_en_interface_source": str(shared.ft_en_path),
                "shared_setup": str(shared.path),
                "shared_setup_sha256": shared.sha256,
                "manifests": shared.manifest_hashes,
                "dataset_a": dataset_a,
                "merge": arm_reports[name],
                "held_out_agreement": arm_agreements[name],
                "runtime_configuration_source": "M/PT_ML",
                "runtime_configuration_exact": True,
                "command": command,
            }
            export.export(
                artifact,
                source=shared.pt_ml_path,
                encoder={key: _as_numpy(value) for key, value in encoder_state.items()},
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
            assert_runtime_config_inherited(reloaded.config, pt_ml.config)

            model = encoder_module.build(reloaded).to(device)
            sanity, profile = direct_runner._forward_check(model, first_mel)
            growth = direct.activation_growth_report(pt_ml_profile, profile)
            bundle = {
                "librispeech": direct_runner._collect_librispeech(
                    model, validation_clips, batch_size=args.batch, eval_frames=eval_frames,
                    device=device, mel_filters=mel_filters, window=window, candidate_name=name,
                ),
                "fleurs": direct_runner._collect_fleurs(
                    model, fleurs_payload, device=device, mel_filters=mel_filters,
                    window=window, candidate_name=name,
                ),
            }
            result = direct_runner._evaluate_stage(
                candidate_name=name, weight=None, stage="pre_quantization",
                candidate_bundle=bundle, reference_bundle=pre_reference,
                manifest_hashes=shared.manifest_hashes, seed=shared.seed,
                comparison=regmean.COMPARISON,
            )
            result_path = stage_root / "results" / name / "pre_quantization.json"
            evaluation.write_result(result_path, result)
            embeddings_path = stage_root / "embeddings" / name / "pre_quantization.safetensors"
            _write_embeddings(
                embeddings_path, bundle, arm=name, stage="pre_quantization",
                plan=arm_plans[name], manifest_hashes=shared.manifest_hashes,
            )
            artifact_report["checks"] = {
                "original_ft_en_interface": interface_equality,
                "export_reload": export_equality,
                "sanity": sanity,
                "activation_growth_vs_pt_ml": growth,
            }
            artifact_report["weight_norm"] = weight_norm
            _write_json(artifact / "regmean_merge.json", artifact_report)
            arm_records[name] = {
                "purpose": ARMS[name],
                "plan": arm_plans[name],
                "artifact": {"path": artifact.relative_to(stage_root).as_posix()},
                "weight_norm": weight_norm,
                "held_out_agreement": arm_agreements[name],
                "forward_check": {"sanity": sanity, "activation_growth_vs_pt_ml": growth},
                "merge_report": _file_record(
                    stage_root / "analysis" / "merges" / f"{name}.json", relative_to=stage_root
                ),
                "result": _file_record(result_path, relative_to=stage_root),
                "embeddings": _file_record(embeddings_path, relative_to=stage_root),
            }
            pre_results[name] = result
            if name == PRIMARY:
                primary_artifact = artifact
            del bundle, model, reloaded, attached, candidate_weights
            arm_states[name] = None
            _release()
        if primary_artifact is None:  # pragma: no cover
            raise ExperimentValidationError("the primary arm produced no artifact")

        # ---- deployment precision for the primary arm ----------------------
        deployment = stage_root / "deployment" / f"{PRIMARY}-Q8_0.gguf"
        baseline_runner._convert_to_q8(primary_artifact, deployment, work)
        primary_pre = load_asr(primary_artifact, mmproj_precision=False)
        actual_post = load_mmproj(deployment, work, config=primary_pre.config)
        simulated_post = load_asr(primary_artifact, mmproj_precision=True)
        simulation_equality = baseline.assert_exact_tensors(simulated_post, actual_post)
        quantization = baseline.quantization_report(primary_pre, actual_post)
        post_model = encoder_module.build(actual_post).to(device)
        post_sanity, post_profile = direct_runner._forward_check(post_model, first_mel)
        post_growth = direct.activation_growth_report(pt_ml_profile, post_profile)
        post_bundle = {
            "librispeech": direct_runner._collect_librispeech(
                post_model, validation_clips, batch_size=args.batch, eval_frames=eval_frames,
                device=device, mel_filters=mel_filters, window=window,
                candidate_name=PRIMARY + "-Q8_0",
            ),
            "fleurs": direct_runner._collect_fleurs(
                post_model, fleurs_payload, device=device, mel_filters=mel_filters,
                window=window, candidate_name=PRIMARY + "-Q8_0",
            ),
        }
        post_reference = direct_runner._read_reference_bundle(
            reference, "post_quantization", fleurs_payload["languages"]
        )
        post_result = direct_runner._evaluate_stage(
            candidate_name=PRIMARY, weight=None, stage="post_quantization",
            candidate_bundle=post_bundle, reference_bundle=post_reference,
            manifest_hashes=shared.manifest_hashes, seed=shared.seed,
            comparison=regmean.COMPARISON,
        )
        evaluation.validate_precision_pair(pre_results[PRIMARY], post_result)
        post_result_path = stage_root / "results" / PRIMARY / "post_quantization.json"
        evaluation.write_result(post_result_path, post_result)
        post_embeddings = stage_root / "embeddings" / PRIMARY / "post_quantization.safetensors"
        _write_embeddings(
            post_embeddings, post_bundle, arm=PRIMARY, stage="post_quantization",
            plan=arm_plans[PRIMARY], manifest_hashes=shared.manifest_hashes,
        )
        precision_delta = baseline.precision_metric_delta(pre_results[PRIMARY], post_result)
        precision_delta_path = stage_root / "results" / PRIMARY / "precision_delta.json"
        _write_json(precision_delta_path, precision_delta)
        parity = baseline_runner._runtime_parity(
            primary_artifact, wav=args.parity_wav, runtime_log=args.runtime_log,
            device=device, work=work,
        )
        arm_records[PRIMARY]["post_quantization"] = {
            "deployment": _file_record(deployment, relative_to=stage_root),
            "actual_artifact_matches_rounding_model": simulation_equality,
            "weight_change": quantization,
            "forward_check": {"sanity": post_sanity, "activation_growth_vs_pt_ml": post_growth},
            "result": _file_record(post_result_path, relative_to=stage_root),
            "embeddings": _file_record(post_embeddings, relative_to=stage_root),
            "precision_delta": _file_record(precision_delta_path, relative_to=stage_root),
            "runtime_parity": parity,
        }
        del post_bundle, post_reference, post_model, actual_post, simulated_post, primary_pre
        _release()

        # ---- against comparisons 1 and 2 -----------------------------------
        delta_records: dict[str, Any] = {}
        for stage, candidate_result in (
            ("pre_quantization", pre_results[PRIMARY]),
            ("post_quantization", post_result),
        ):
            table = regmean.delta_table(
                candidate_result, reference.results[stage], task_arithmetic.get(stage)
            )
            json_path = stage_root / "analysis" / f"delta-{stage}.json"
            markdown_path = stage_root / "analysis" / f"delta-{stage}.md"
            _write_json(json_path, table)
            _write_delta_markdown(markdown_path, table)
            delta_records[stage] = {
                "json": _file_record(json_path, relative_to=stage_root),
                "table": _file_record(markdown_path, relative_to=stage_root),
            }

        for name, record in arm_records.items():
            record["artifact"]["files"] = _tree_records(stage_root / record["artifact"]["path"])

        run_report = {
            "schema_version": "1.0",
            "comparison": regmean.COMPARISON,
            "artifact_kind": regmean.ARTIFACT_KIND,
            "status": "complete",
            "command": command,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "torch": torch.__version__,
                "device": str(device),
                "working_directory": str(REPOSITORY),
                "deterministic_algorithms": True,
                "tf32": False,
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
            "pt_en_role": (
                "invariant check only; RegMean merges full weights and has no shared base"
            ),
            "paired_pt_ml_reference": {
                "run": _file_record(reference.run_path),
                "embeddings": _file_record(reference.embeddings_path),
                "results": {
                    stage: _file_record(path) for stage, path in reference.result_paths.items()
                },
                "exact_frozen_arrays_used": True,
            },
            "task_arithmetic_reference": (
                {
                    "run": _file_record(args.task_arithmetic / "run.json"),
                    "candidate_id": direct.candidate_id(1.0),
                }
                if args.task_arithmetic is not None
                else None
            ),
            "runtime_reader": runtime_reader,
            "dataset_a": dataset_a,
            "merge": {
                "method": "RegMean++ (Nguyen et al., TMLR 2026), Algorithm 1",
                "gradient_descent": False,
                "precision": "F32 weights, F64 solves, F32/F64 Gram accumulation",
                "alpha_grid": _file_record(
                    stage_root / "analysis" / "alpha_grid.json", relative_to=stage_root
                ),
                "selected_alpha": alpha,
                "arms": _file_record(
                    stage_root / "analysis" / "arms.json", relative_to=stage_root
                ),
                "ood_gram_ablation": ood_provenance,
            },
            "evaluation": {
                "split": "LibriSpeech/validation and frozen FLEURS",
                "english_frame_cap": eval_frames,
                "reserved_librispeech_test_encoded": False,
                "every_arm_pre_quantization": True,
                "primary_arm_post_quantization": True,
                "against_comparisons_1_and_2": delta_records,
            },
            "candidates": arm_records,
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
    logger.info("Comparison 6 complete: %s", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shared-setup", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="the completed Comparison 1 output directory")
    parser.add_argument("--task-arithmetic", type=Path, default=None,
                        help="the completed Comparison 2 output directory, for the delta table")
    parser.add_argument("--slurp", type=Path, required=True)
    parser.add_argument("--speech-massive", type=Path, required=True)
    parser.add_argument("--ood-speech-massive", type=Path, default=None,
                        help="an off-domain multilingual Gram manifest for the ablation")
    parser.add_argument("--alpha", type=float, default=None,
                        help="reuse a shrinkage already selected on the frozen held-out "
                             "split instead of running the grid; requires --alpha-source")
    parser.add_argument("--alpha-source", default=None,
                        help="the run whose held-out selection --alpha comes from")
    parser.add_argument("--arm", action="append", default=None,
                        help="restrict the reference arms built; the primary is always built")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--gram-batch", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parity-wav", type=Path, default=None)
    parser.add_argument("--runtime-log", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if (args.alpha is None) != (args.alpha_source is None):
        parser.error("--alpha and --alpha-source go together")
    if args.alpha is not None and args.alpha not in regmean.ALPHAS:
        parser.error(f"--alpha must come from the frozen grid {list(regmean.ALPHAS)}")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()
