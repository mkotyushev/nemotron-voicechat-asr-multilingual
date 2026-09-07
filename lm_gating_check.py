#!/usr/bin/env python3
"""Run the blocking checks that gate comparison 7 (`REGMEAN_INTERFACE_DESIGN.md` §11).

Stages are separate commands so that an expensive generation pass is written
once, immutably, and can be rescored without asking the language model again:

    dimensions  check 2, from the encoder configurations alone
    sample      freeze the MASSIVE utterances the gate is decided on
    generate    check 1, one teacher path over the frozen sample
    gap         check 3, the text-path versus audio-path target gap
    score       checks 1 and 4, plus the gate verdicts, from the raw passes

The perception path drives the pinned deployment container and needs nothing
else running.  The text path needs a stock llama.cpp server on the same GGUF,
started with the checkpoint's real end-of-turn token, because the GGUF declares
`</s>` while its chat format ends a turn with `<SPECIAL_12>`:

    llama-server -m nemotron_voicechat_11b-stt-llm-Q8_0.gguf \
      --override-kv tokenizer.ggml.eos_token_id=int:12 \
      -ngl 36 -c 8192 -np 4 --host 127.0.0.1 --port 9099

Run one path at a time: each holds its own copy of an 11B model, and one GPU
does not have room for both alongside the served runtime.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from asr_align import gating
from asr_align.experiments import ExperimentValidationError, sha256_file, stable_json_sha256
from asr_align.voice_assistant import file_provenance, write_json_once

CONTAINER_TMP = "/tmp/lm-gating"


def _command() -> str:
    return " ".join(shlex.quote(argument) for argument in sys.argv)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_run(output: Path, stage: str, payload: Mapping[str, Any] | None = None) -> None:
    write_json_once(
        output / f"run-{stage}.json",
        {"command": _command(), "utc": _now(), "stage": stage, **(payload or {})},
    )


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], header: Mapping[str, Any]) -> None:
    if path.exists():
        raise ExperimentValidationError(f"refusing to replace an immutable raw pass: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"header": dict(header)}, ensure_ascii=False, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ExperimentValidationError(f"{path}: empty raw pass")
    header = json.loads(lines[0]).get("header")
    if not isinstance(header, Mapping):
        raise ExperimentValidationError(f"{path}: first line is not a header")
    return dict(header), [json.loads(line) for line in lines[1:]]


# --------------------------------------------------------------- container


def _docker_put(container: str, data: bytes, destination: str) -> None:
    """Copy bytes into the container's tmpfs, which `docker cp` cannot reach."""

    subprocess.run(
        ["docker", "exec", "-i", container, "sh", "-c", f"mkdir -p {CONTAINER_TMP} && cat > {destination}"],
        input=data,
        check=True,
    )


def _silence_wav(seconds: float, rate: int = 16_000) -> bytes:
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


def _container_provenance(container: str, model: str, mmproj: str) -> dict[str, Any]:
    image = subprocess.run(
        ["docker", "inspect", "--format", "{{.Config.Image}}", container],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    digests = subprocess.run(
        ["docker", "exec", container, "sha256sum", model, mmproj, "/app/llama-voicechat"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return {
        "container": container,
        "image": image,
        "files": {line.split()[1]: line.split()[0] for line in digests},
    }


# -------------------------------------------------------------- commands


def _dimensions(args: argparse.Namespace) -> int:
    configs = {}
    for entry in args.config:
        name, _, path = entry.partition("=")
        if not path:
            raise ExperimentValidationError(f"--config takes NAME=PATH, got {entry!r}")
        configs[name] = Path(path)
    report = gating.confirm_feed_forward_width(configs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json_once(args.output / "dimensions.json", report)
    _write_run(args.output, "dimensions")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _sample(args: argparse.Namespace) -> int:
    sample = gating.build_massive_sample(
        args.massive,
        count=args.count,
        partition=args.partition,
        seed=args.seed,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = gating.write_sample(args.output / "sample.json", sample)
    _write_run(args.output, "sample")
    print(json.dumps({
        "sample": str((args.output / "sample.json").resolve()),
        "manifest_sha256": frozen["manifest_sha256"],
        "utterances": len(frozen["utterances"]),
        "languages": frozen["languages"],
    }, indent=2))
    return 0


def _generate(args: argparse.Namespace) -> int:
    sample = gating.load_sample(args.sample)
    destination = args.output / f"generations-{args.path}.jsonl"
    if destination.exists():
        raise ExperimentValidationError(f"refusing to replace an immutable raw pass: {destination}")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.path == "perception":
        silence = _silence_wav(args.silence_seconds)
        _docker_put(args.container, silence, f"{CONTAINER_TMP}/silence.wav")
        engine = gating.PerceptionPathEngine(
            container=args.container,
            model=args.model,
            mmproj=args.mmproj,
            silence=f"{CONTAINER_TMP}/silence.wav",
            n_gpu_layers=args.n_gpu_layers,
            threads=args.threads,
            extra_decoding_seconds=args.extra_decoding_seconds,
            session_seconds=args.session_seconds,
        )
        engine_record: dict[str, Any] = {
            "kind": "pinned voicechat CLI, --serve",
            "arguments": engine.arguments,
            "silence_seconds": args.silence_seconds,
            "silence_sha256": stable_json_sha256({"wav": silence.hex()}),
            **_container_provenance(args.container, args.model, args.mmproj),
        }
    else:
        engine = gating.TextPathEngine(args.endpoint, max_tokens=args.max_tokens)
        engine_record = {
            "kind": "stock llama.cpp server, rendered chat format",
            "endpoint": args.endpoint,
            "max_tokens": args.max_tokens,
            "properties": engine.properties(),
        }
        if args.llama_binary is not None:
            engine_record["binary"] = file_provenance(args.llama_binary)
        if args.text_server_command:
            # /props does not report how the server was started, and this
            # checkpoint needs an end-of-turn override to stop at all: the GGUF
            # declares `</s>` while the chat format ends a turn with
            # `<SPECIAL_12>`, token 12.  Without it every reply runs to the
            # token cap.
            engine_record["launch_command"] = args.text_server_command

    languages = tuple(args.language) if args.language else tuple(sample["languages"])
    header = {
        "schema_version": gating.RAW_SCHEMA_VERSION,
        "stage": "generate",
        "path": args.path,
        "command": _command(),
        "utc": _now(),
        "sample": {
            "path": str(Path(args.sample).resolve()),
            "manifest_sha256": sample["manifest_sha256"],
        },
        "languages": list(languages),
        "prompts": {key: gating.SYSTEM_PROMPTS[key] for key in args.prompt},
        "framing": (
            gating.PERCEPTION_FRAMING if args.path == "perception" else gating.TEXT_FRAMING
        ),
        "limit": args.limit,
        "engine": engine_record,
    }

    done = 0
    total = len(args.prompt) * len(languages) * (args.limit or len(sample["utterances"]))

    def progress(_row: Mapping[str, Any]) -> None:
        nonlocal done
        done += 1
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}", file=sys.stderr, flush=True)

    try:
        rows = gating.generate_replies(
            engine,
            sample,
            path=args.path,
            languages=languages,
            prompts=tuple(args.prompt),
            limit=args.limit,
            progress=progress,
        )
    finally:
        engine.close()

    _write_rows(destination, rows, header)
    print(json.dumps({"raw": str(destination.resolve()), "rows": len(rows)}, indent=2))
    return 0


def _english_clips(
    root: Path, revision: str, count: int, exclude: Sequence[str], seed: int = 0
) -> list[dict[str, Any]]:
    """Draw English assistant clips from the pinned BFCL audio shards.

    The frozen speech-to-action pilot is drawn from the same shards and is
    untouched by invariant 8, so its case ids are excluded here rather than
    reused.  The draw is seeded over the sorted pool of both shards, because
    taking the head of a lexicographic sort would draw one category only.
    """

    import random

    import pyarrow.parquet as parquet

    excluded = set(exclude)
    clips: list[dict[str, Any]] = []
    for category in ("simple", "multiple"):
        path = Path(root).resolve() / f"BFCL_v3_{category}" / "test-00000-of-00001.parquet"
        if not path.is_file():
            raise ExperimentValidationError(f"missing BFCL audio shard: {path}")
        table = parquet.read_table(path, columns=["id", "question", "audio"])
        digest = sha256_file(path)
        for index, identifier in enumerate(table.column("id").to_pylist()):
            identifier = str(identifier)
            if identifier in excluded:
                continue
            question = json.loads(table.column("question")[index].as_py())
            if not isinstance(question, list) or len(question) != 1:
                continue
            audio = table.column("audio")[index].as_py()
            payload = audio.get("bytes") if isinstance(audio, Mapping) else None
            if not payload:
                continue
            clips.append(
                {
                    "id": identifier,
                    "shard": {"path": str(path), "sha256": digest, "revision": revision},
                    "transcript": str(question[0]),
                    "bytes": payload,
                }
            )
    clips.sort(key=lambda clip: clip["id"])
    if len(clips) < count:
        raise ExperimentValidationError(f"only {len(clips)} usable BFCL clips, need {count}")
    chosen = random.Random(seed).sample(range(len(clips)), count)
    return [clips[index] for index in sorted(chosen)]


def _gap(args: argparse.Namespace) -> int:
    destination = args.output / f"generations-gap-{'-'.join(sorted(args.source))}.jsonl"
    if destination.exists():
        raise ExperimentValidationError(f"refusing to replace an immutable raw pass: {destination}")
    args.output.mkdir(parents=True, exist_ok=True)

    clips = _english_clips(
        args.bfcl_audio, args.bfcl_audio_revision, args.count, args.exclude, args.seed
    )
    system = gating.SYSTEM_PROMPTS[args.prompt]
    sources = tuple(args.source)
    rows: list[dict[str, Any]] = []
    engine_record: dict[str, Any] = {}

    # The two engines are kept in separate invocations on purpose: they hold
    # their own copy of the language model, and one GPU does not have room for
    # both alongside the served runtime.
    cli_sources = [name for name in sources if name in {"audio", "text_perception"}]
    if cli_sources:
        silence = _silence_wav(args.silence_seconds)
        _docker_put(args.container, silence, f"{CONTAINER_TMP}/silence.wav")
        for clip in clips:
            _docker_put(args.container, clip["bytes"], f"{CONTAINER_TMP}/{clip['id']}.wav")
        engine = gating.PerceptionPathEngine(
            container=args.container,
            model=args.model,
            mmproj=args.mmproj,
            silence=f"{CONTAINER_TMP}/silence.wav",
            n_gpu_layers=args.n_gpu_layers,
            threads=args.threads,
            extra_decoding_seconds=args.extra_decoding_seconds,
            session_seconds=args.session_seconds,
        )
        engine_record = {
            "arguments": engine.arguments,
            **_container_provenance(args.container, args.model, args.mmproj),
        }
        try:
            for clip in clips:
                if "audio" in cli_sources:
                    # B1's target: the original VoiceChat reading the clip itself.
                    audio = engine.generate(system, audio=f"{CONTAINER_TMP}/{clip['id']}.wav")
                    rows.append({
                        "id": clip["id"], "source": "audio", "transcript": clip["transcript"],
                        "reply": audio["text"], "seconds": audio["seconds"],
                        "diagnostics": {
                            k: v for k, v in audio.items() if k not in {"text", "seconds"}
                        },
                    })
                if "text_perception" in cli_sources:
                    # B2's procedure applied to English: the same model reading
                    # the transcript on the perception channel instead of
                    # hearing it.
                    text = engine.generate(
                        gating.render_prompt("perception", system, clip["transcript"])
                    )
                    rows.append({
                        "id": clip["id"], "source": "text_perception",
                        "transcript": clip["transcript"],
                        "reply": text["text"], "seconds": text["seconds"],
                        "diagnostics": {
                            k: v for k, v in text.items() if k not in {"text", "seconds"}
                        },
                    })
        finally:
            engine.close()

    if "text_chat" in sources:
        if not args.endpoint:
            raise ExperimentValidationError("the text_chat source needs --endpoint")
        text_engine = gating.TextPathEngine(args.endpoint, max_tokens=args.max_tokens)
        engine_record = {
            "endpoint": args.endpoint,
            "properties": text_engine.properties(),
            **engine_record,
        }
        for clip in clips:
            reply = text_engine.generate(
                gating.render_prompt("text", system, clip["transcript"])
            )
            rows.append({
                "id": clip["id"], "source": "text_chat", "transcript": clip["transcript"],
                "reply": reply["text"], "seconds": reply["seconds"],
                "diagnostics": {k: v for k, v in reply.items() if k not in {"text", "seconds"}},
            })

    header = {
        "schema_version": gating.RAW_SCHEMA_VERSION,
        "stage": "gap",
        "command": _command(),
        "utc": _now(),
        "prompt_id": args.prompt,
        "system_prompt": system,
        "clips": [
            {k: v for k, v in clip.items() if k != "bytes"} | {"bytes": len(clip["bytes"])}
            for clip in clips
        ],
        "excluded_pilot_ids": list(args.exclude),
        "seed": args.seed,
        "sources": list(sources),
        "engine": engine_record,
    }
    _write_rows(destination, rows, header)
    print(json.dumps({"raw": str(destination.resolve()), "rows": len(rows)}, indent=2))
    return 0


def _score(args: argparse.Namespace) -> int:
    identifier, identifier_report = gating.fit_language_identifier(args.massive)

    passes: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in args.generations:
        header, pass_rows = _read_rows(path)
        passes.append({"file": file_provenance(path), "header": header})
        rows.extend(pass_rows)
    scored = gating.score_rows(rows, identifier) if rows else {"cells": {}, "rows": []}
    verdicts = gating.gate_verdicts(scored["cells"])

    gap: dict[str, Any] | None = None
    gap_rows: list[dict[str, Any]] = []
    for path in args.gap:
        header, rows_for_pass = _read_rows(path)
        passes.append({"file": file_provenance(path), "header": header})
        gap_rows.extend(rows_for_pass)
    if gap_rows:
        gap = gating.score_target_gap(gap_rows, identifier)

    dimensions = None
    if args.dimensions is not None:
        dimensions = json.loads(Path(args.dimensions).read_text(encoding="utf-8"))

    result: dict[str, Any] = {
        "schema_version": gating.RESULT_SCHEMA_VERSION,
        "command": _command(),
        "utc": _now(),
        "design_record": "REGMEAN_INTERFACE_DESIGN.md section 11",
        "raw_passes": passes,
        "system_prompts": dict(gating.SYSTEM_PROMPTS),
        "gate_thresholds": dict(gating.GATE_THRESHOLDS),
        "language_identifier": identifier_report,
        "check_1_and_4": {"cells": scored["cells"], "verdicts": verdicts},
        "check_2_dimensions": dimensions,
        "check_3_target_gap": None if gap is None else gap["summary"],
    }
    result["result_sha256"] = stable_json_sha256(result)

    args.output.mkdir(parents=True, exist_ok=True)
    write_json_once(args.output / "result.json", result)
    if scored["rows"]:
        _write_rows(
            args.output / "scored-replies.jsonl",
            scored["rows"],
            {"schema_version": gating.RESULT_SCHEMA_VERSION, "stage": "score"},
        )
    if gap is not None:
        _write_rows(
            args.output / "scored-gap.jsonl",
            gap["pairs"],
            {"schema_version": gating.RESULT_SCHEMA_VERSION, "stage": "gap"},
        )
    print(json.dumps({
        "result": str((args.output / "result.json").resolve()),
        "result_sha256": result["result_sha256"],
        "verdicts": {key: value["passes"] for key, value in sorted(verdicts.items())},
    }, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dimensions = subparsers.add_parser("dimensions", help="check 2, from the configurations")
    dimensions.add_argument("--config", action="append", required=True, metavar="NAME=PATH")
    dimensions.add_argument("--output", type=Path, required=True)

    sample = subparsers.add_parser("sample", help="freeze the MASSIVE gating sample")
    sample.add_argument("--massive", type=Path, required=True, help="MASSIVE 1.1 data directory")
    sample.add_argument("--count", type=int, default=100)
    sample.add_argument("--partition", default="dev")
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--output", type=Path, required=True)

    generate = subparsers.add_parser("generate", help="check 1, one teacher path")
    generate.add_argument("--sample", type=Path, required=True)
    generate.add_argument("--path", choices=gating.PATHS, required=True)
    generate.add_argument("--language", action="append")
    generate.add_argument(
        "--prompt", action="append", choices=sorted(gating.SYSTEM_PROMPTS), default=None
    )
    generate.add_argument("--limit", type=int)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--container", default="nemotron-voicechat")
    generate.add_argument("--model", default="/models/nemotron_voicechat_11b-stt-llm-Q8_0.gguf")
    generate.add_argument("--mmproj", default="/models/mmproj-voicechat-perception-Q8_0.gguf")
    generate.add_argument("--n-gpu-layers", type=int, default=24)
    generate.add_argument("--threads", type=int, default=12)
    generate.add_argument("--extra-decoding-seconds", type=float, default=12.0)
    generate.add_argument("--session-seconds", type=float, default=60.0)
    generate.add_argument("--silence-seconds", type=float, default=1.0)
    generate.add_argument("--endpoint", default="http://127.0.0.1:9099")
    generate.add_argument("--llama-binary", type=Path)
    generate.add_argument("--max-tokens", type=int, default=200)
    generate.add_argument(
        "--text-server-command",
        help="the command the text-path server was started with, recorded verbatim",
    )

    gap = subparsers.add_parser("gap", help="check 3, text-path versus audio-path targets")
    gap.add_argument("--bfcl-audio", type=Path, required=True)
    gap.add_argument("--bfcl-audio-revision", required=True)
    gap.add_argument("--count", type=int, default=24)
    gap.add_argument("--exclude", action="append", default=[])
    gap.add_argument("--seed", type=int, default=0)
    gap.add_argument("--prompt", choices=sorted(gating.SYSTEM_PROMPTS), default="A_english_only")
    gap.add_argument("--output", type=Path, required=True)
    gap.add_argument("--container", default="nemotron-voicechat")
    gap.add_argument("--model", default="/models/nemotron_voicechat_11b-stt-llm-Q8_0.gguf")
    gap.add_argument("--mmproj", default="/models/mmproj-voicechat-perception-Q8_0.gguf")
    gap.add_argument("--n-gpu-layers", type=int, default=24)
    gap.add_argument("--threads", type=int, default=12)
    gap.add_argument("--extra-decoding-seconds", type=float, default=12.0)
    gap.add_argument("--session-seconds", type=float, default=60.0)
    gap.add_argument("--silence-seconds", type=float, default=1.0)
    gap.add_argument("--endpoint", default="")
    gap.add_argument(
        "--source",
        action="append",
        choices=["audio", "text_perception", "text_chat"],
        default=None,
        help="which target procedures to run; repeat, or omit for all three",
    )
    gap.add_argument("--max-tokens", type=int, default=200)

    score = subparsers.add_parser("score", help="checks 1 and 4, and the gate verdicts")
    score.add_argument("--generations", type=Path, action="append", default=[])
    score.add_argument("--gap", type=Path, action="append", default=[])
    score.add_argument("--dimensions", type=Path)
    score.add_argument("--massive", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "generate" and not args.prompt:
        args.prompt = sorted(gating.SYSTEM_PROMPTS)
    if args.command == "gap" and not args.source:
        args.source = ["audio", "text_perception", "text_chat"]
    handlers = {
        "dimensions": _dimensions,
        "sample": _sample,
        "generate": _generate,
        "gap": _gap,
        "score": _score,
    }
    try:
        return handlers[args.command](args)
    except ExperimentValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
