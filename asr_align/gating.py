"""The blocking checks that gate comparison 7 (`REGMEAN_INTERFACE_DESIGN.md` §11).

Comparison 7 distils a teacher into a trained ``proj``.  Three of its four
premises are unverified assumptions about the frozen language model rather than
about the encoder, and one is a configuration fact that sizes comparison 6's
Gram collection:

1. the frozen language model reads fr/de/ru text and answers in-language;
2. ``n_ff`` is ``intermediate_size``, which fixes the largest linear input
   dimension and therefore the Dataset A frame budget;
3. the text-path and audio-path targets are on a comparable scale;
4. teacher quality is known, so the distillation ceiling is known.

A capability the teacher lacks cannot be distilled, so these run before
anything is built.  Nothing here fits, selects or exports a model: the module
generates from a frozen language model under two frozen system prompts and
scores what came back.

Two teacher paths are measured, because "text-only" is ambiguous for a model
whose only input is a perception channel:

``perception``
    The pinned deployment binary (``llama-voicechat --serve``), with the
    utterance carried in the system prompt.  This is the channel VoiceChat was
    trained to read text on -- one token per 12.5 Hz timeline frame, summed
    with the previous frame's token embedding -- and a turn is triggered by a
    silent clip so that no audio content reaches the model.  It is the path the
    deployment runtime actually has.

``text``
    The base Nemotron-H chat format the checkpoint inherits, rendered
    explicitly and completed by a stock llama.cpp server.  This is the ordinary
    reading of "run the language model text-only", and it is a different input
    mode: no perception offset, no silent clip.

They disagree, so the check reports both rather than picking one.  The served
bridge cannot host either: ``render_system_prompt`` in the runtime strips
non-ASCII, which erases Cyrillic entirely, so the teacher has to be driven
through the CLI or a text server and never through ``/v1/realtime``.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .experiments import ExperimentValidationError, sha256_file, stable_json_sha256
from .manifests import write_frozen

SAMPLE_SCHEMA_VERSION = "lm-gating-sample-1.0"
RAW_SCHEMA_VERSION = "lm-gating-raw-1.0"
RESULT_SCHEMA_VERSION = "lm-gating-result-1.0"

LANGUAGES = ("en", "fr", "de", "ru")
FOREIGN_LANGUAGES = ("fr", "de", "ru")
LOCALES = {"en": "en-US", "fr": "fr-FR", "de": "de-DE", "ru": "ru-RU"}
PATHS = ("perception", "text")

# The two output-language conditions of the design record, frozen.  They differ
# in exactly one clause so that a difference in the replies is attributable.
# Under invariant 9 these are two comparison rows, never one row with a prompt
# column, and they are ASCII because the deployment runtime requires it.
SYSTEM_PROMPTS: dict[str, str] = {
    "A_english_only": (
        "You are a voice assistant. Always answer the user in English, even when "
        "the user speaks another language. Answer briefly."
    ),
    "B_input_language": (
        "You are a voice assistant. Always answer the user in the same language "
        "the user spoke, and in no other language. Answer briefly."
    ),
}

# The teacher passes a condition when it obeys the condition often enough for
# its output to be worth distilling.  Declared here rather than chosen after
# seeing the numbers.
GATE_THRESHOLDS = {"A_english_only": 0.90, "B_input_language": 0.80}

# The framing that carries the utterance on each path.  Recorded with the
# result: a gate verdict is only meaningful against the framing that produced
# it, and these were selected on a handful of pilot utterances before the
# sample was drawn.
PERCEPTION_FRAMING = '{system}\n\nThe user said: "{utterance}"'
TEXT_FRAMING = (
    "<SPECIAL_10>System\n{system}\n"
    "<SPECIAL_11>User\n{utterance}\n"
    "<SPECIAL_11>Assistant\n<think></think>"
)


# --------------------------------------------------------------- check 2


def confirm_feed_forward_width(configs: Mapping[str, Path]) -> dict[str, Any]:
    """Check 2: confirm ``n_ff`` and derive comparison 6's Gram frame budget.

    `asr_align.encoder` reads ``n_ff`` straight from ``intermediate_size``, so
    the design record's open question is whether that value is the 4x`n_embd`
    it assumed.  The answer sets the largest input dimension any RegMean solve
    faces and therefore how much audio Dataset A needs: the paper's own default
    is about four rows per input dimension, and copying its literal sample count
    to speech would solve the widest layers at half that.
    """

    if not configs:
        raise ExperimentValidationError("no encoder configurations to confirm")
    checkpoints: dict[str, Any] = {}
    widths: set[int] = set()
    hidden: set[int] = set()
    for name, path in sorted(configs.items()):
        path = Path(path).resolve()
        if not path.is_file():
            raise ExperimentValidationError(f"missing encoder configuration: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
        encoder = config.get("encoder_config")
        if not isinstance(encoder, Mapping):
            raise ExperimentValidationError(f"{path}: no encoder_config")
        for key in ("intermediate_size", "hidden_size", "num_mel_bins",
                    "subsampling_conv_channels", "num_hidden_layers"):
            if not isinstance(encoder.get(key), int):
                raise ExperimentValidationError(f"{path}: encoder_config lacks int {key}")
        widths.add(int(encoder["intermediate_size"]))
        hidden.add(int(encoder["hidden_size"]))
        checkpoints[name] = {
            "config": {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            },
            "intermediate_size": int(encoder["intermediate_size"]),
            "hidden_size": int(encoder["hidden_size"]),
            "num_hidden_layers": int(encoder["num_hidden_layers"]),
            "num_mel_bins": int(encoder["num_mel_bins"]),
            "subsampling_conv_channels": int(encoder["subsampling_conv_channels"]),
            "sliding_window": encoder.get("sliding_window"),
        }
    if len(widths) != 1 or len(hidden) != 1:
        raise ExperimentValidationError(
            f"candidates disagree on encoder width: intermediate {sorted(widths)}, "
            f"hidden {sorted(hidden)}"
        )
    n_ff = widths.pop()
    n_embd = hidden.pop()

    # `asr_align.encoder`: 128 mel bins survive three stride-2 convolutions that
    # pad 2 left and 1 right on both axes, leaving 17 frequency bins, and
    # `subsampling.linear` flattens channel-major over them.
    mel = {value["num_mel_bins"] for value in checkpoints.values()}.pop()
    channels = {value["subsampling_conv_channels"] for value in checkpoints.values()}.pop()
    freq = mel
    for _ in range(3):
        freq = (freq + 2 + 1 - 3) // 2 + 1
    subsampling_in = channels * freq

    dimensions = {
        "self_attn.{q,k,v,o}_proj": n_embd,
        "feed_forward{1,2}.linear1": n_embd,
        "feed_forward{1,2}.linear2": n_ff,
        "conv.pointwise_conv{1,2}": n_embd,
        "subsampling.linear": subsampling_in,
    }
    largest_name, largest = max(dimensions.items(), key=lambda item: item[1])
    frames = 4 * largest
    seconds = frames / 12.5
    return {
        "check": "feed_forward_width",
        "checkpoints": checkpoints,
        "n_embd": n_embd,
        "n_ff": n_ff,
        "n_ff_is_intermediate_size": True,
        "n_ff_over_n_embd": n_ff / n_embd,
        "linear_input_dimensions": dimensions,
        "largest_linear_input": {"tensor": largest_name, "d_in": largest},
        "gram_frames_at_4x": frames,
        "gram_seconds_at_4x": seconds,
        "gram_minutes_at_4x": seconds / 60.0,
        "frame_rate_hz": 12.5,
        "note": (
            "Dataset A needs at least gram_frames_at_4x encoder output frames per "
            "candidate, equalized across candidates and each G normalized by its "
            "frame count; the paper's literal 256 samples would solve the widest "
            "layers at about half this ratio"
        ),
    }


# --------------------------------------------------------------- the sample


def _read_massive(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExperimentValidationError(f"{path}:{number}: invalid JSON") from exc
            rows.append(row)
    if not rows:
        raise ExperimentValidationError(f"{path}: no utterances")
    return rows


def build_massive_sample(
    root: Path,
    *,
    count: int = 100,
    partition: str = "dev",
    seed: int = 0,
    languages: Sequence[str] = FOREIGN_LANGUAGES,
) -> dict[str, Any]:
    """Freeze the utterances the gate is decided on.

    MASSIVE localized rather than translated, so the same id is a *different*
    request in each locale; the parallel ``en-US`` row is kept as a scoring
    reference only, never as a target (design record §7).  Only ids present in
    every locale are eligible, so one seeded draw serves all of them and the
    English control row is the same request the foreign rows are.

    The draw is from ``dev``, which `REGMEAN_INTERFACE_DESIGN.md` §13 allocates
    to Dataset A and B2.  The ids are recorded so that the utterances the gate
    was decided on can be excluded from Dataset B2 later.
    """

    root = Path(root).resolve()
    if count < 1:
        raise ExperimentValidationError("the gating sample needs at least one utterance")
    wanted = tuple(dict.fromkeys(languages))
    for language in wanted:
        if language not in LOCALES:
            raise ExperimentValidationError(f"unsupported gating language: {language}")
    needed = tuple(dict.fromkeys(("en",) + wanted))

    files: dict[str, Any] = {}
    by_language: dict[str, dict[str, dict[str, Any]]] = {}
    for language in needed:
        path = root / f"{LOCALES[language]}.jsonl"
        if not path.is_file():
            raise ExperimentValidationError(f"missing MASSIVE locale file: {path}")
        files[language] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        rows = {}
        for row in _read_massive(path):
            if row.get("partition") != partition:
                continue
            identifier = str(row.get("id"))
            if identifier in rows:
                raise ExperimentValidationError(f"{path}: duplicate id {identifier}")
            rows[identifier] = row
        if not rows:
            raise ExperimentValidationError(f"{path}: partition {partition!r} is empty")
        by_language[language] = rows

    shared = set.intersection(*(set(rows) for rows in by_language.values()))
    if len(shared) < count:
        raise ExperimentValidationError(
            f"only {len(shared)} ids are present in every locale, need {count}"
        )
    # A seeded draw over a numerically sorted pool: the same seed and pool give
    # the same utterances on any machine, and the pool order does not depend on
    # the order the files happen to list ids in.
    import random

    pool = sorted(shared, key=lambda value: (int(value) if value.isdigit() else math.inf, value))
    chosen = sorted(
        random.Random(seed).sample(pool, count),
        key=lambda value: (int(value) if value.isdigit() else math.inf, value),
    )

    utterances = []
    for identifier in chosen:
        record: dict[str, Any] = {"id": identifier, "locales": {}}
        for language in needed:
            row = by_language[language][identifier]
            record["locales"][language] = {
                "locale": LOCALES[language],
                "utt": str(row["utt"]),
                "annot_utt": str(row.get("annot_utt", "")),
                # §7: the replacement method identifies where the two output
                # conditions are most likely to diverge, so it is recorded per
                # utterance even though it no longer acts as a filter.
                "slot_method": row.get("slot_method") or [],
                "judgment_language_identification": sorted(
                    {
                        str(judgment.get("language_identification"))
                        for judgment in row.get("judgments") or []
                        if isinstance(judgment, Mapping)
                    }
                ),
            }
        first = by_language[needed[0]][identifier]
        record["intent"] = str(first["intent"])
        record["scenario"] = str(first["scenario"])
        utterances.append(record)

    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "dataset": {
            "name": "MASSIVE",
            "version": "1.1",
            "repository": "https://github.com/alexa/massive",
            "download": (
                "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
                "amazon-massive-dataset-1.1.tar.gz"
            ),
            "license": "CC BY 4.0",
            "files": files,
        },
        "partition": partition,
        "seed": seed,
        "count": count,
        "languages": list(wanted),
        "reference_language": "en",
        "system_prompts": dict(SYSTEM_PROMPTS),
        "utterances": utterances,
    }


def load_sample(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = value.get("manifest_sha256")
    unsigned = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if not isinstance(recorded, str) or stable_json_sha256(unsigned) != recorded:
        raise ExperimentValidationError(f"{path}: gating sample digest mismatch")
    if value.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ExperimentValidationError(
            f"{path}: unsupported gating sample schema {value.get('schema_version')!r}"
        )
    return value


def write_sample(path: Path, sample: Mapping[str, Any]) -> dict[str, Any]:
    return write_frozen(Path(path), sample)


# ------------------------------------------------------- language identity


def _profile_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)).strip()


def _ngrams(text: str, order: int) -> Iterable[str]:
    padded = f" {text} "
    for index in range(len(padded) - order + 1):
        yield padded[index : index + order]


class LanguageIdentifier:
    """A character n-gram naive Bayes over the four MASSIVE locales.

    Output-language identification is the cheap, discrete half of comparison
    7's readout (§12) and the gate's own primary statistic, so it cannot depend
    on an unpinned external model.  MASSIVE ships tens of thousands of
    utterances per locale in exactly the register the teacher answers in, so
    the classifier is fitted from the frozen corpus itself and its held-out
    accuracy is reported next to every rate it produces.
    """

    def __init__(self, order: int = 3, smoothing: float = 0.5):
        self.order = int(order)
        self.smoothing = float(smoothing)
        self.log_prior: dict[str, float] = {}
        self.log_likelihood: dict[str, dict[str, float]] = {}
        self.log_unseen: dict[str, float] = {}
        self.vocabulary = 0

    def fit(self, corpus: Mapping[str, Sequence[str]]) -> "LanguageIdentifier":
        counts: dict[str, Counter[str]] = {}
        documents: dict[str, int] = {}
        vocabulary: set[str] = set()
        for language, texts in sorted(corpus.items()):
            counter: Counter[str] = Counter()
            for text in texts:
                counter.update(_ngrams(_profile_text(text), self.order))
            if not counter:
                raise ExperimentValidationError(f"no training text for {language}")
            counts[language] = counter
            documents[language] = len(texts)
            vocabulary |= set(counter)
        self.vocabulary = len(vocabulary)
        total_documents = sum(documents.values())
        for language, counter in counts.items():
            total = sum(counter.values()) + self.smoothing * self.vocabulary
            self.log_prior[language] = math.log(documents[language] / total_documents)
            self.log_likelihood[language] = {
                gram: math.log((value + self.smoothing) / total)
                for gram, value in counter.items()
            }
            self.log_unseen[language] = math.log(self.smoothing / total)
        return self

    def scores(self, text: str) -> dict[str, float]:
        grams = list(_ngrams(_profile_text(text), self.order))
        result = {}
        for language, prior in self.log_prior.items():
            unseen = self.log_unseen[language]
            table = self.log_likelihood[language]
            result[language] = prior + sum(table.get(gram, unseen) for gram in grams)
        return result

    def identify(self, text: str) -> tuple[str | None, float]:
        """Return the most likely language and its margin over the runner-up.

        Empty or punctuation-only text has no language; a margin in nats per
        character keeps short replies from looking as confident as long ones.
        """

        cleaned = _profile_text(text)
        if not cleaned:
            return None, 0.0
        scores = sorted(self.scores(text).items(), key=lambda item: item[1], reverse=True)
        best, runner_up = scores[0], scores[1]
        return best[0], (best[1] - runner_up[1]) / max(len(cleaned), 1)


def fit_language_identifier(
    root: Path,
    *,
    languages: Sequence[str] = LANGUAGES,
    train_partition: str = "train",
    test_partition: str = "test",
    order: int = 3,
    max_per_language: int = 4000,
) -> tuple[LanguageIdentifier, dict[str, Any]]:
    """Fit on MASSIVE ``train`` and report accuracy on the untouched ``test``."""

    root = Path(root).resolve()
    train: dict[str, list[str]] = {}
    test: dict[str, list[str]] = {}
    for language in languages:
        rows = _read_massive(root / f"{LOCALES[language]}.jsonl")
        train[language] = [
            str(row["utt"]) for row in rows if row.get("partition") == train_partition
        ][:max_per_language]
        test[language] = [
            str(row["utt"]) for row in rows if row.get("partition") == test_partition
        ]
    identifier = LanguageIdentifier(order=order).fit(train)

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for language, texts in test.items():
        for text in texts:
            predicted, _ = identifier.identify(text)
            confusion[language][str(predicted)] += 1
    per_language = {
        language: {
            "n": sum(counter.values()),
            "accuracy": counter[language] / max(sum(counter.values()), 1),
            "confusion": dict(sorted(counter.items())),
        }
        for language, counter in sorted(confusion.items())
    }
    total = sum(value["n"] for value in per_language.values())
    correct = sum(value["accuracy"] * value["n"] for value in per_language.values())
    report = {
        "model": "character naive Bayes",
        "order": order,
        "train_partition": train_partition,
        "train_utterances_per_language": {k: len(v) for k, v in sorted(train.items())},
        "held_out_partition": test_partition,
        "held_out_accuracy": correct / max(total, 1),
        "per_language": per_language,
    }
    return identifier, report


# ------------------------------------------------------------- the engines


class PerceptionPathEngine:
    """The pinned deployment binary in ``--serve`` mode, one process per run.

    The utterance rides in the system prompt, which the CLI tokenizes onto the
    perception channel one token per frame, and a silent clip triggers the turn
    so nothing but the text conditions the reply.  ``reset`` clears the
    conversation and re-arms the system prompt, which is what makes a single
    process enough for the whole sample: the model is loaded once.
    """

    def __init__(
        self,
        *,
        container: str,
        model: str,
        mmproj: str,
        silence: str,
        n_gpu_layers: int = 24,
        threads: int = 12,
        extra_decoding_seconds: float = 12.0,
        session_seconds: float = 60.0,
        ready_timeout: float = 900.0,
        turn_timeout: float = 600.0,
    ):
        self.arguments = [
            "docker", "exec", "-i", container,
            "/app/llama-voicechat",
            "-m", model,
            "--mmproj", mmproj,
            "--serve",
            "--temp", "0",
            "-ngl", str(n_gpu_layers),
            "-t", str(threads),
            "--extra-decoding-seconds", str(extra_decoding_seconds),
            "--session-seconds", str(session_seconds),
        ]
        self.silence = silence
        self.turn_timeout = turn_timeout
        self.errors: list[dict[str, Any]] = []
        self.process = subprocess.Popen(
            self.arguments,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._await("ready", ready_timeout)

    def _send(self, command: Mapping[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def _await(self, kind: str, timeout: float) -> dict[str, Any]:
        assert self.process.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if not line:
                raise ExperimentValidationError(
                    f"the voicechat CLI exited before emitting {kind!r}"
                )
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # a log line that reached stdout
            if event.get("kind") == "error":
                self.errors.append(dict(event))
                continue
            if event.get("kind") == kind:
                return event
        raise ExperimentValidationError(f"timed out waiting for {kind!r} from the voicechat CLI")

    def generate(self, prompt: str, *, audio: str | None = None) -> dict[str, Any]:
        before = len(self.errors)
        self._send({"cmd": "system", "text": prompt})
        self._await("system", self.turn_timeout)
        started = time.monotonic()
        self._send({"cmd": "turn", "audio": audio or self.silence})
        turn = self._await("turn_end", self.turn_timeout)
        self._send({"cmd": "reset"})
        self._await("reset", self.turn_timeout)
        return {
            "text": str(turn.get("text", "")),
            "frames": turn.get("frames"),
            "spoken": turn.get("spoken"),
            "tool_calls": turn.get("tool_calls"),
            "failed": bool(turn.get("failed", False)),
            "seconds": round(time.monotonic() - started, 3),
            "errors": self.errors[before:],
        }

    def close(self) -> None:
        try:
            self._send({"cmd": "quit"})
        except (BrokenPipeError, ValueError, AssertionError):
            pass
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.process.kill()


class TextPathEngine:
    """A stock llama.cpp server completing the checkpoint's inherited chat format.

    The prompt is rendered here rather than by the server's template engine so
    that the exact conditioning text is recordable and identical to what the
    perception path is asked, apart from the framing this module declares.
    """

    def __init__(self, endpoint: str, *, timeout: float = 600.0, max_tokens: int = 200):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens

    def properties(self) -> dict[str, Any]:
        with urllib.request.urlopen(f"{self.endpoint}/props", timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "model_path": payload.get("model_path"),
            "n_ctx": payload.get("n_ctx"),
            "build_info": payload.get("build_info"),
        }

    def generate(self, prompt: str, **_: Any) -> dict[str, Any]:
        body = json.dumps(
            {
                "prompt": prompt,
                "temperature": 0.0,
                "n_predict": self.max_tokens,
                "cache_prompt": True,
                # `<SPECIAL_12>` is this checkpoint's end of turn, but the GGUF
                # declares `</s>`; the server is started with an eos override,
                # and these are the belt to that pair of braces.
                "stop": ["<SPECIAL_11>", "<SPECIAL_10>", "<SPECIAL_12>"],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ExperimentValidationError(f"text-path server request failed: {exc}") from exc
        return {
            "text": str(payload.get("content", "")),
            "tokens": payload.get("tokens_predicted"),
            "stop_type": payload.get("stop_type"),
            "seconds": round(time.monotonic() - started, 3),
            "errors": [],
        }

    def close(self) -> None:  # symmetry with the perception path
        return None


# ------------------------------------------------------------- generation


def render_prompt(path: str, system: str, utterance: str) -> str:
    if path == "perception":
        return PERCEPTION_FRAMING.format(system=system, utterance=utterance)
    if path == "text":
        return TEXT_FRAMING.format(system=system, utterance=utterance)
    raise ExperimentValidationError(f"unknown teacher path: {path!r}")


def generate_replies(
    engine: Any,
    sample: Mapping[str, Any],
    *,
    path: str,
    languages: Sequence[str],
    prompts: Sequence[str] = tuple(SYSTEM_PROMPTS),
    limit: int | None = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Run one teacher path over the sample; one record per cell, in order."""

    rows: list[dict[str, Any]] = []
    utterances = list(sample["utterances"])[: limit or len(sample["utterances"])]
    for prompt_id in prompts:
        system = SYSTEM_PROMPTS[prompt_id]
        for language in languages:
            for utterance in utterances:
                locale = utterance["locales"].get(language)
                if locale is None:
                    raise ExperimentValidationError(
                        f"utterance {utterance['id']} has no {language} locale"
                    )
                text = locale["utt"]
                rendered = render_prompt(path, system, text)
                reply = engine.generate(rendered)
                rows.append(
                    {
                        "path": path,
                        "prompt_id": prompt_id,
                        "language": language,
                        "id": utterance["id"],
                        "intent": utterance["intent"],
                        "utterance": text,
                        "prompt_sha256": stable_json_sha256({"prompt": rendered}),
                        "reply": reply["text"],
                        "seconds": reply.get("seconds"),
                        "diagnostics": {
                            key: value
                            for key, value in reply.items()
                            if key not in {"text", "seconds"}
                        },
                    }
                )
                if progress is not None:
                    progress(rows[-1])
    return rows


# --------------------------------------------------------------- scoring


_TURN_MARKERS = ("</s>", "<s>", "<SPECIAL_10>", "<SPECIAL_11>", "<SPECIAL_12>", "<think>",
                 "</think>")


def clean_reply(text: str) -> str:
    """Strip the channel markers that are turn structure, not content.

    The perception path's text channel opens with the previous turn's ``</s>``
    and the forced ``<s>``; neither is anything the model said.
    """

    cleaned = text
    for marker in _TURN_MARKERS:
        cleaned = cleaned.replace(marker, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _words(text: str) -> list[str]:
    return _profile_text(text).split()


def is_echo(reply: str, utterance: str) -> bool:
    """Whether the reply mostly repeats the request instead of answering it."""

    request = _words(utterance)
    answer = _words(reply)
    if not request or not answer:
        return False
    joined = " ".join(answer)
    if " ".join(request) in joined:
        return True
    shared = sum((Counter(request) & Counter(answer)).values())
    return shared / len(request) >= 0.8 and len(answer) <= 2 * len(request)


def is_looping(reply: str, *, min_repeats: int = 3, max_period: int = 6) -> bool:
    """Whether the tail of the reply is a repeating cluster.

    The deployed CLI cuts a turn on the same condition; a looping reply is a
    failure to answer, not a long answer, and it must not be counted as one.
    """

    words = _words(reply)
    for period in range(1, max_period + 1):
        if len(words) < period * min_repeats:
            continue
        tail = words[-period * min_repeats :]
        if all(
            tail[index] == tail[index % period]
            for index in range(len(tail))
        ):
            return True
    return False


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    identifier: LanguageIdentifier,
    *,
    margin: float = 0.05,
) -> dict[str, Any]:
    """Turn raw replies into the per-cell rates the gate is decided on."""

    scored: list[dict[str, Any]] = []
    for row in rows:
        cleaned = clean_reply(str(row["reply"]))
        predicted, confidence = identifier.identify(cleaned)
        if confidence < margin:
            predicted = None
        looping = is_looping(cleaned)
        echoing = is_echo(cleaned, str(row["utterance"]))
        scored.append(
            {
                **{key: row[key] for key in ("path", "prompt_id", "language", "id", "intent")},
                "utterance": row["utterance"],
                "reply": cleaned,
                "raw_reply": row["reply"],
                "words": len(_words(cleaned)),
                "empty": not cleaned,
                "looping": looping,
                "echoing": echoing,
                "identified_language": predicted,
                "identification_margin": round(confidence, 4),
                # A usable target answers, in one identifiable language, without
                # looping or parroting the request back.
                "usable": bool(cleaned and not looping and not echoing and predicted),
                "seconds": row.get("seconds"),
            }
        )

    cells: dict[str, Any] = {}
    for row in scored:
        key = f"{row['path']}|{row['prompt_id']}|{row['language']}"
        cells.setdefault(key, []).append(row)

    summary: dict[str, Any] = {}
    for key, group in sorted(cells.items()):
        path, prompt_id, language = key.split("|")
        n = len(group)
        answered = [row for row in group if not row["empty"]]
        expected = "en" if prompt_id == "A_english_only" else language
        obeyed = [row for row in group if row["identified_language"] == expected]
        usable_obeyed = [row for row in obeyed if row["usable"]]
        summary[key] = {
            "path": path,
            "prompt_id": prompt_id,
            "input_language": language,
            "expected_output_language": expected,
            "n": n,
            "answered_rate": len(answered) / n,
            "looping_rate": sum(row["looping"] for row in group) / n,
            "echoing_rate": sum(row["echoing"] for row in group) / n,
            "expected_language_rate": len(obeyed) / n,
            "english_output_rate": sum(
                row["identified_language"] == "en" for row in group
            ) / n,
            "input_language_output_rate": sum(
                row["identified_language"] == language for row in group
            ) / n,
            "unidentified_rate": sum(row["identified_language"] is None for row in group) / n,
            # What survives the §11.4 gate: obeys the condition and is not a
            # loop or an echo.  This is the distillation ceiling for the cell.
            "usable_target_rate": len(usable_obeyed) / n,
            "median_words": _median([row["words"] for row in group]),
            "language_histogram": dict(
                sorted(Counter(str(row["identified_language"]) for row in group).items())
            ),
        }
    return {"cells": summary, "rows": scored}


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def gate_verdicts(cells: Mapping[str, Any]) -> dict[str, Any]:
    """Decide, per path and condition, whether comparison 7 has a teacher.

    A condition passes on a path when every foreign input language reaches its
    declared threshold on the rate that condition is defined by.  Condition A
    is graded on English output, condition B on in-language output, both after
    the usability gate, because a looping reply in the right language is not a
    target.
    """

    verdicts: dict[str, Any] = {}
    for path in sorted({value["path"] for value in cells.values()}):
        for prompt_id, threshold in sorted(GATE_THRESHOLDS.items()):
            foreign = [
                value
                for value in cells.values()
                if value["path"] == path
                and value["prompt_id"] == prompt_id
                and value["input_language"] != "en"
            ]
            if not foreign:
                continue
            rates = {
                value["input_language"]: value["usable_target_rate"] for value in foreign
            }
            worst = min(rates.values())
            verdicts[f"{path}|{prompt_id}"] = {
                "path": path,
                "prompt_id": prompt_id,
                "threshold": threshold,
                "usable_target_rate_by_language": dict(sorted(rates.items())),
                "worst_language_rate": worst,
                "passes": worst >= threshold,
            }
    return verdicts


# ------------------------------------------------------- check 3, the gap


def token_f1(reference: str, candidate: str) -> float:
    """Unigram F1 between two replies, the scale-free half of the gap."""

    left, right = Counter(_words(reference)), Counter(_words(candidate))
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(right.values())
    recall = overlap / sum(left.values())
    return 2 * precision * recall / (precision + recall)


def score_target_gap(
    rows: Sequence[Mapping[str, Any]], identifier: LanguageIdentifier
) -> dict[str, Any]:
    """Check 3: how far the text-path target is from the audio-path target.

    B1's target is what the original VoiceChat emits on the clip and B2's is
    what the language model writes from a transcript.  If the two procedures
    produce systematically different lengths or contents, their losses are not
    on one scale and any weighting between the terms is arbitrary.
    """

    by_clip: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_clip.setdefault(str(row["id"]), {})[str(row["source"])] = row

    comparisons: list[dict[str, Any]] = []
    for identifier_key, sources in sorted(by_clip.items()):
        audio = sources.get("audio")
        if audio is None:
            continue
        audio_text = clean_reply(str(audio["reply"]))
        for name, row in sorted(sources.items()):
            if name == "audio":
                continue
            text = clean_reply(str(row["reply"]))
            comparisons.append(
                {
                    "id": identifier_key,
                    "text_source": name,
                    "audio_words": len(_words(audio_text)),
                    "text_words": len(_words(text)),
                    "token_f1": round(token_f1(audio_text, text), 4),
                    "audio_language": identifier.identify(audio_text)[0],
                    "text_language": identifier.identify(text)[0],
                    "audio_reply": audio_text,
                    "text_reply": text,
                }
            )

    summary: dict[str, Any] = {}
    for name in sorted({row["text_source"] for row in comparisons}):
        group = [row for row in comparisons if row["text_source"] == name]
        audio_words = [row["audio_words"] for row in group]
        text_words = [row["text_words"] for row in group]
        summary[name] = {
            "n": len(group),
            "median_token_f1": _median([row["token_f1"] for row in group]),
            "mean_token_f1": sum(row["token_f1"] for row in group) / max(len(group), 1),
            "median_audio_words": _median(audio_words),
            "median_text_words": _median(text_words),
            "length_ratio_text_over_audio": (
                _median(text_words) / _median(audio_words) if _median(audio_words) else None
            ),
            "same_language_rate": sum(
                row["audio_language"] == row["text_language"] for row in group
            ) / max(len(group), 1),
            "audio_empty_rate": sum(not row["audio_words"] for row in group) / max(len(group), 1),
        }
    return {"pairs": comparisons, "summary": summary}
