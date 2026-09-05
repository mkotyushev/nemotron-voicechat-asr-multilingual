#!/usr/bin/env python3
"""Prepare and run the paired pre-TTS VoiceChat tool-calling pilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from pathlib import Path

from asr_align import voice_assistant
from asr_align.experiments import ExperimentValidationError, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze the paired synthetic pilot")
    prepare.add_argument("--rfcb", type=Path, required=True)
    prepare.add_argument("--silero", type=Path, required=True)
    english = prepare.add_mutually_exclusive_group(required=True)
    english.add_argument("--english-model", type=Path)
    english.add_argument("--bfcl-audio", type=Path)
    prepare.add_argument("--bfcl-audio-revision")
    prepare.add_argument("--russian-model", type=Path, required=True)
    prepare.add_argument("--english-speaker", default="en_0")
    prepare.add_argument("--russian-speaker", default="xenia")
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="run a deployed candidate and score it")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--endpoint", default="ws://127.0.0.1:9070/v1/realtime")
    run.add_argument("--expected-asr-model", required=True)
    run.add_argument("--candidate-id", required=True)
    role = run.add_mutually_exclusive_group(required=True)
    role.add_argument(
        "--comparison",
        type=int,
        choices=range(1, 6),
        help="score a candidate produced by one of comparisons 1-5",
    )
    role.add_argument(
        "--control",
        help="score a reference encoder, such as the original VoiceChat perception encoder",
    )
    run.add_argument(
        "--precision",
        choices=voice_assistant.PRECISIONS,
        required=True,
    )
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument(
        "--shared-setup",
        type=Path,
        help="frozen shared_setup.json; required for a comparison row",
    )
    run.add_argument("--runtime-repository", type=Path, required=True)
    run.add_argument("--runtime-revision", required=True)
    run.add_argument(
        "--runtime-env",
        type=Path,
        required=True,
        help="deployment environment file the pinned server was started with",
    )
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--response-budget",
        type=float,
        default=30.0,
        help="seconds allowed for a response, measured from the end of the audio",
    )
    run.add_argument("--pace", type=float, default=1.0)
    run.add_argument("--settle", type=float, default=2.0)
    run.add_argument("--inter-case-delay", type=float, default=0.5)
    run.add_argument("--bootstrap-samples", type=int, default=2000)
    run.add_argument("--seed", type=int, default=0)

    score = subparsers.add_parser("score", help="rescore an immutable raw run")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--raw", type=Path, required=True)
    score.add_argument(
        "--output", type=Path, required=True, help="directory for the rescored result"
    )
    score.add_argument("--bootstrap-samples", type=int, default=2000)
    score.add_argument("--seed", type=int, default=0)

    compare = subparsers.add_parser(
        "compare", help="collect scored rows into one comparable table"
    )
    compare.add_argument("--result", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def _command() -> str:
    return " ".join(shlex.quote(argument) for argument in sys.argv)


def _prepare(args: argparse.Namespace) -> int:
    manifest = voice_assistant.prepare_rfcb_pilot(
        rfcb=args.rfcb,
        silero=args.silero,
        english_model=args.english_model,
        russian_model=args.russian_model,
        output=args.output,
        bfcl_audio=args.bfcl_audio,
        bfcl_audio_revision=args.bfcl_audio_revision,
        english_speaker=args.english_speaker,
        russian_speaker=args.russian_speaker,
    )
    print(json.dumps({
        "manifest": str((args.output / "manifest.json").resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "cases": len(manifest["cases"]),
    }, indent=2))
    return 0


def _run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise ExperimentValidationError(f"run output already exists: {output}")
    manifest = voice_assistant.load_manifest(args.manifest)
    if args.comparison is not None and args.shared_setup is None:
        raise ExperimentValidationError(
            "a comparison row must pin the frozen shared_setup.json with --shared-setup"
        )
    candidate = {
        "role": "comparison" if args.comparison is not None else "control",
        "comparison": args.comparison,
        "candidate_id": args.candidate_id,
        "precision": args.precision,
        "artifact": voice_assistant.file_provenance(args.artifact),
        "runtime": {
            "repository": str(args.runtime_repository.resolve()),
            "revision": args.runtime_revision,
            "environment": voice_assistant.read_runtime_environment(args.runtime_env),
        },
    }
    if args.control is not None:
        candidate["control"] = args.control
        candidate.pop("comparison")
    if args.shared_setup is not None:
        candidate["shared_setup"] = voice_assistant.file_provenance(args.shared_setup)
    voice_assistant.validate_candidate(candidate)
    raw = asyncio.run(
        voice_assistant.run_realtime(
            manifest=manifest,
            endpoint=args.endpoint,
            candidate=candidate,
            expected_asr_model=args.expected_asr_model,
            response_budget=args.response_budget,
            pace=args.pace,
            settle=args.settle,
            inter_case_delay=args.inter_case_delay,
        )
    )
    result = voice_assistant.evaluate_raw(
        manifest,
        raw,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output.mkdir(parents=True)
    voice_assistant.write_json_once(output / "raw.json", raw)
    voice_assistant.write_json_once(output / "result.json", result)
    run_record = {
        "command": _command(),
        "manifest": voice_assistant.file_provenance(args.manifest),
        "raw": voice_assistant.file_provenance(output / "raw.json"),
        "result": voice_assistant.file_provenance(output / "result.json"),
        "candidate": candidate,
    }
    voice_assistant.write_json_once(output / "run.json", run_record)
    primary = {
        language: result["per_language"][language]["single_call_exact_accuracy"]["rate"]
        for language in voice_assistant.LANGUAGES
    }
    print(json.dumps({
        "output": str(output),
        "primary": primary,
        "paired": result["paired"],
    }, indent=2))
    return 0


def _compare(args: argparse.Namespace) -> int:
    results = []
    for path in args.result:
        value = json.loads(path.read_text(encoding="utf-8"))
        voice_assistant.validate_result(value)
        results.append(value)
    table = voice_assistant.build_comparison(results)
    output = args.output.resolve()
    voice_assistant.write_json_once(output / "comparison.json", table)
    markdown = voice_assistant.render_comparison_markdown(table)
    path = output / "comparison.md"
    if path.exists() and path.read_text(encoding="utf-8") != markdown:
        raise ExperimentValidationError(f"refusing to replace immutable table {path}")
    path.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


def _score(args: argparse.Namespace) -> int:
    manifest = voice_assistant.load_manifest(args.manifest)
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    result = voice_assistant.evaluate_raw(
        manifest,
        raw,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    voice_assistant.write_json_once(output / "result.json", result)
    # A rescored result is only usable if it names the run it came from.
    voice_assistant.write_json_once(output / "score.json", {
        "command": _command(),
        "manifest": voice_assistant.file_provenance(args.manifest),
        "raw": voice_assistant.file_provenance(args.raw),
        "result": voice_assistant.file_provenance(output / "result.json"),
        "schema_version": voice_assistant.RESULT_SCHEMA_VERSION,
    })
    print(json.dumps({
        "output": str(output),
        "sha256": sha256_file(output / "result.json"),
    }, indent=2))
    return 0


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "run":
            return _run(args)
        if args.command == "compare":
            return _compare(args)
        return _score(args)
    except (ExperimentValidationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
