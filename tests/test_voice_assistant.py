from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from asr_align import voice_assistant
from asr_align.experiments import ExperimentValidationError, stable_json_sha256


def _signed(payload: dict, field: str) -> dict:
    value = copy.deepcopy(payload)
    value[field] = stable_json_sha256(value)
    return value


def _case() -> dict:
    return {
        "semantic_id": "simple_1",
        "category": "simple",
        "tools": [{
            "type": "function",
            "name": "math.factorial",
            "description": "Calculate factorial.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        }],
        "ground_truth": [{"math.factorial": {"number": [5]}}],
        "tool_output": {"factorial": 120},
        "response_facts": [["120"]],
        "audio": {
            "en": {
                "path": "audio/simple_1.en.wav", "transcript": "Factorial of five.",
                "sample_rate": 24000, "frames": 24000, "channels": 1,
                "bytes": 10, "sha256": "0" * 64, "synthesis": {},
            },
            "ru": {
                "path": "audio/simple_1.ru.wav", "transcript": "Факториал пяти.",
                "sample_rate": 24000, "frames": 24000, "channels": 1,
                "bytes": 10, "sha256": "1" * 64, "synthesis": {},
            },
        },
    }


def _manifest() -> dict:
    return _signed({
        "schema_version": voice_assistant.MANIFEST_SCHEMA_VERSION,
        "dataset": "test",
        "root": "/tmp/test",
        "split": "development",
        "languages": ["en", "ru"],
        "sample_rate": 24000,
        "system_prompt": voice_assistant.SYSTEM_PROMPT,
        "cases": [_case()],
    }, "manifest_sha256")


def _candidate(**overrides: object) -> dict:
    candidate = {
        "role": "comparison",
        "comparison": 1,
        "candidate_id": "baseline",
        "precision": "post_quantization",
        "artifact": {"path": "/models/baseline.gguf", "bytes": 1, "sha256": "a" * 64},
        "shared_setup": {"path": "/setup.json", "bytes": 1, "sha256": "b" * 64},
        "runtime": {
            "repository": "/runtime",
            "revision": "c" * 40,
            "environment": {
                "file": {"path": "/runtime/.env", "bytes": 1, "sha256": "d" * 64},
                "values": {"VC_REF": "e" * 40, "TEMP": "0.0"},
            },
        },
    }
    candidate.update(overrides)
    return candidate


def _raw(
    manifest: dict,
    en_call: bool = True,
    ru_call: bool = True,
    candidate: dict | None = None,
    response_budget: float = 30.0,
) -> dict:
    records = []
    for language, called in (("en", en_call), ("ru", ru_call)):
        records.append({
            "semantic_id": "simple_1",
            "language": language,
            "status": "completed",
            "before_tts_text": "The result is 120.",
            "tool_calls": ([{"name": "math.factorial", "arguments": '{"number":5}'}]
                           if called else []),
        })
    return _signed({
        "schema_version": voice_assistant.RAW_SCHEMA_VERSION,
        "candidate": candidate or _candidate(),
        "manifest_sha256": manifest["manifest_sha256"],
        "system_prompt_sha256": stable_json_sha256(manifest["system_prompt"]),
        "timing": {
            "response_budget_seconds": response_budget,
            "budget_measured_from": "end_of_streamed_audio",
            "realtime_pace": 1.0,
            "settle_seconds": 2.0,
            "inter_case_delay_seconds": 0.5,
        },
        "records": records,
    }, "raw_sha256")


class VoiceAssistantManifestTests(unittest.TestCase):
    def test_manifest_requires_distinct_complete_pairs_and_ascii_prompt(self) -> None:
        manifest = _manifest()
        voice_assistant.validate_manifest(manifest)
        bad = copy.deepcopy(manifest)
        bad["cases"][0]["audio"]["ru"]["sha256"] = "0" * 64
        bad.pop("manifest_sha256")
        bad = _signed(bad, "manifest_sha256")
        with self.assertRaisesRegex(ExperimentValidationError, "identical"):
            voice_assistant.validate_manifest(bad)

        bad = copy.deepcopy(manifest)
        bad["system_prompt"] = "Отвечай по-английски"
        bad.pop("manifest_sha256")
        bad = _signed(bad, "manifest_sha256")
        with self.assertRaisesRegex(ExperimentValidationError, "ASCII"):
            voice_assistant.validate_manifest(bad)

    def test_write_json_once_rejects_changed_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            voice_assistant.write_json_once(path, {"a": 1})
            voice_assistant.write_json_once(path, {"a": 1})
            with self.assertRaisesRegex(ExperimentValidationError, "immutable"):
                voice_assistant.write_json_once(path, {"a": 2})


class VoiceAssistantScoringTests(unittest.TestCase):
    def test_typed_single_call_scoring_accepts_alternative_omission(self) -> None:
        case = _case()
        score = voice_assistant.score_tool_calls(
            [{"name": "math.factorial", "arguments": '{"number": 5}'}],
            case["ground_truth"],
            case["tools"],
        )
        self.assertTrue(score["single_call_exact"])

        wrong_type = voice_assistant.score_tool_calls(
            [{"name": "math.factorial", "arguments": '{"number": "5"}'}],
            case["ground_truth"],
            case["tools"],
        )
        self.assertFalse(wrong_type["arguments_correct"])
        self.assertIn("types", " ".join(wrong_type["errors"]))

    def test_missing_calls_do_not_count_as_cross_language_agreement(self) -> None:
        manifest = _manifest()
        result = voice_assistant.evaluate_raw(
            manifest, _raw(manifest, en_call=False, ru_call=False), bootstrap_samples=20
        )
        self.assertEqual(
            result["paired"]["same_nonempty_tool_call_rate"]["rate"], 0.0
        )
        self.assertEqual(
            result["per_language"]["ru"]["single_call_exact_accuracy"]["rate"], 0.0
        )

    def test_result_separates_text_tool_and_language_metrics(self) -> None:
        manifest = _manifest()
        result = voice_assistant.evaluate_raw(manifest, _raw(manifest), bootstrap_samples=20)
        self.assertEqual(
            result["per_language"]["en"]["single_call_exact_accuracy"]["rate"], 1.0
        )
        self.assertEqual(result["paired"]["before_tts_text_token_f1"]["mean"], 1.0)
        self.assertEqual(
            result["per_language"]["ru"]["english_output_compliance_rate"]["rate"], 1.0
        )

    def test_raw_must_cover_each_frozen_language_once(self) -> None:
        manifest = _manifest()
        raw = _raw(manifest)
        raw["records"].pop()
        raw.pop("raw_sha256")
        raw = _signed(raw, "raw_sha256")
        with self.assertRaisesRegex(ExperimentValidationError, "exactly cover"):
            voice_assistant.validate_raw(raw, manifest=manifest)

    def test_raw_must_record_a_budget_measured_from_the_end_of_the_audio(self) -> None:
        manifest = _manifest()
        raw = _raw(manifest)
        raw["timing"]["budget_measured_from"] = "session_start"
        raw.pop("raw_sha256")
        raw = _signed(raw, "raw_sha256")
        with self.assertRaisesRegex(ExperimentValidationError, "end of the audio"):
            voice_assistant.validate_raw(raw, manifest=manifest)


class VoiceAssistantCandidateTests(unittest.TestCase):
    def test_comparison_row_must_pin_a_shared_setup(self) -> None:
        candidate = _candidate()
        candidate.pop("shared_setup")
        with self.assertRaisesRegex(ExperimentValidationError, "shared_setup"):
            voice_assistant.validate_candidate(candidate)

    def test_control_row_names_what_it_controls_for_and_needs_no_setup(self) -> None:
        candidate = _candidate(role="control", control="original FT_EN encoder")
        candidate.pop("comparison")
        candidate.pop("shared_setup")
        voice_assistant.validate_candidate(candidate)

        anonymous = _candidate(role="control")
        anonymous.pop("comparison")
        with self.assertRaisesRegex(ExperimentValidationError, "controls for"):
            voice_assistant.validate_candidate(anonymous)

    def test_runtime_environment_must_pin_generation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("# comment\nVC_REF=" + "f" * 40 + "\nTEMP=0.0\n", encoding="utf-8")
            recorded = voice_assistant.read_runtime_environment(path)
            self.assertEqual(recorded["values"]["TEMP"], "0.0")

            path.write_text("VC_REF=" + "f" * 40 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ExperimentValidationError, "TEMP"):
                voice_assistant.read_runtime_environment(path)


class VoiceAssistantFactScoringTests(unittest.TestCase):
    def test_spoken_numbers_satisfy_digit_facts(self) -> None:
        score = voice_assistant._fact_score(
            "The factorial of five is one hundred and twenty.", [["120"]]
        )
        self.assertTrue(score["all_facts_present"])
        self.assertTrue(score["spoken_numbers_normalized"])

    def test_a_conjunction_between_numbers_does_not_add_them(self) -> None:
        self.assertEqual(
            voice_assistant._spoken_numbers_to_digits("forty and fifty is ten"),
            "40 and 50 is 10",
        )

    def test_a_bare_number_fact_must_be_its_own_token(self) -> None:
        self.assertFalse(
            voice_assistant._fact_score("the answer is one hundred and twenty", [["1"]])[
                "all_facts_present"
            ]
        )

    def test_decimals_survive_normalization(self) -> None:
        self.assertEqual(
            voice_assistant._spoken_numbers_to_digits(
                voice_assistant._normalized_text("eight point eight five farads.")
            ),
            "8.85 farads",
        )

    def test_separator_variant_is_wrong_but_distinguishable(self) -> None:
        case = _case()
        score = voice_assistant.score_tool_calls(
            [{"name": "math_factorial", "arguments": '{"number": 5}'}],
            case["ground_truth"],
            case["tools"],
        )
        self.assertFalse(score["single_call_exact"])
        self.assertTrue(score["tool_name_separator_variant"])
        self.assertIn("different separator", " ".join(score["errors"]))

        wrong_tool = voice_assistant.score_tool_calls(
            [{"name": "math.gcd", "arguments": '{"number": 5}'}],
            case["ground_truth"],
            case["tools"],
        )
        self.assertFalse(wrong_tool["tool_name_separator_variant"])


class VoiceAssistantComparisonTests(unittest.TestCase):
    def _result(self, **overrides: object) -> dict:
        manifest = _manifest()
        return voice_assistant.evaluate_raw(
            manifest, _raw(manifest, **overrides), bootstrap_samples=20
        )

    def test_table_orders_controls_first_and_keeps_both_languages(self) -> None:
        control = _candidate(
            role="control", candidate_id="ft-en", control="original FT_EN encoder"
        )
        control.pop("comparison")
        table = voice_assistant.build_comparison([
            self._result(),
            self._result(candidate=control, ru_call=False),
        ])
        self.assertEqual([row["candidate_id"] for row in table["rows"]], ["ft-en", "baseline"])
        self.assertEqual(table["rows"][0]["ru"]["single_call_exact_accuracy"], 0.0)
        self.assertEqual(table["rows"][1]["ru"]["single_call_exact_accuracy"], 1.0)
        self.assertIn("ft-en", voice_assistant.render_comparison_markdown(table))

    def test_table_refuses_rows_scored_under_different_conditions(self) -> None:
        with self.assertRaisesRegex(ExperimentValidationError, "response_budget_seconds"):
            voice_assistant.build_comparison([
                self._result(),
                self._result(
                    candidate=_candidate(candidate_id="other", comparison=2),
                    response_budget=10.0,
                ),
            ])

    def test_table_refuses_the_same_candidate_twice(self) -> None:
        with self.assertRaisesRegex(ExperimentValidationError, "duplicate candidate"):
            voice_assistant.build_comparison([self._result(), self._result()])


if __name__ == "__main__":
    unittest.main()
