from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from asr_align import gating
from asr_align.experiments import ExperimentValidationError


def _encoder_config(intermediate: int = 4096, hidden: int = 1024) -> dict:
    return {
        "encoder_config": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": 24,
            "num_mel_bins": 128,
            "subsampling_conv_channels": 256,
            "sliding_window": 57,
        }
    }


def _write_config(root: Path, name: str, payload: dict) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _massive_rows(locale: str, word: str, count: int = 12) -> list[dict]:
    rows = []
    for index in range(count):
        partition = ("train", "dev", "test")[index % 3]
        rows.append(
            {
                "id": str(index),
                "locale": locale,
                "partition": partition,
                "scenario": "iot",
                "intent": "iot_hue_lightoff",
                "utt": f"{word} {word} number {index}",
                "annot_utt": f"{word} [number : {index}]",
                "slot_method": [{"slot": "number", "method": "translation"}],
                "judgments": [{"language_identification": "target"}],
            }
        )
    return rows


def _write_massive(root: Path, words: dict[str, str], count: int = 12) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for language, word in words.items():
        path = root / f"{gating.LOCALES[language]}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in _massive_rows(gating.LOCALES[language], word, count)),
            encoding="utf-8",
        )
    return root


class FeedForwardWidthTests(unittest.TestCase):
    def test_the_widest_linear_is_the_subsampling_flatten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = gating.confirm_feed_forward_width(
                {
                    "PT_ML": _write_config(root, "ml", _encoder_config()),
                    "PT_EN": _write_config(root, "en", _encoder_config()),
                }
            )
        self.assertEqual(report["n_ff"], 4096)
        self.assertTrue(report["n_ff_is_intermediate_size"])
        # 128 mel bins survive three stride-2 convolutions as 17, times 256
        # channels, which is what asr_align.encoder builds Linear(4352, 1024)
        # from; it is wider than n_ff and therefore sets the frame budget.
        self.assertEqual(report["largest_linear_input"]["d_in"], 4352)
        self.assertEqual(report["largest_linear_input"]["tensor"], "subsampling.linear")
        self.assertEqual(report["gram_frames_at_4x"], 4 * 4352)
        self.assertAlmostEqual(report["gram_minutes_at_4x"], 4 * 4352 / 12.5 / 60.0)

    def test_candidates_that_disagree_on_width_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ExperimentValidationError):
                gating.confirm_feed_forward_width(
                    {
                        "PT_ML": _write_config(root, "ml", _encoder_config(4096)),
                        "PT_EN": _write_config(root, "en", _encoder_config(2048)),
                    }
                )

    def test_a_configuration_without_an_encoder_block_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ExperimentValidationError):
                gating.confirm_feed_forward_width({"X": _write_config(root, "x", {})})


class SampleTests(unittest.TestCase):
    def test_the_draw_is_deterministic_and_carries_every_locale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _write_massive(
                Path(directory) / "data", {"en": "light", "fr": "lumiere", "de": "licht", "ru": "свет"}
            )
            first = gating.build_massive_sample(root, count=3, partition="dev", seed=0)
            second = gating.build_massive_sample(root, count=3, partition="dev", seed=0)
            other = gating.build_massive_sample(root, count=3, partition="dev", seed=1)
        self.assertEqual(first["utterances"], second["utterances"])
        self.assertNotEqual(
            [row["id"] for row in first["utterances"]],
            [row["id"] for row in other["utterances"]],
        )
        for row in first["utterances"]:
            self.assertEqual(sorted(row["locales"]), ["de", "en", "fr", "ru"])
            self.assertTrue(row["locales"]["ru"]["slot_method"])
        self.assertEqual(first["system_prompts"], gating.SYSTEM_PROMPTS)

    def test_a_sample_larger_than_the_shared_pool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _write_massive(
                Path(directory) / "data", {"en": "light", "fr": "lumiere", "de": "licht", "ru": "свет"}
            )
            with self.assertRaises(ExperimentValidationError):
                gating.build_massive_sample(root, count=99, partition="dev", seed=0)

    def test_a_frozen_sample_round_trips_and_a_tampered_one_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = _write_massive(
                Path(directory) / "data", {"en": "light", "fr": "lumiere", "de": "licht", "ru": "свет"}
            )
            path = Path(directory) / "sample.json"
            gating.write_sample(path, gating.build_massive_sample(root, count=3, seed=0))
            loaded = gating.load_sample(path)
            self.assertEqual(len(loaded["utterances"]), 3)

            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["utterances"][0]["locales"]["ru"]["utt"] = "другое"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ExperimentValidationError):
                gating.load_sample(path)


class LanguageIdentifierTests(unittest.TestCase):
    def test_the_classifier_separates_scripts_and_reports_no_language_for_empty_text(self) -> None:
        identifier = gating.LanguageIdentifier(order=3).fit(
            {
                "en": ["turn off the lights please", "what is the weather today"],
                "ru": ["выключи свет пожалуйста", "какая сегодня погода"],
            }
        )
        self.assertEqual(identifier.identify("please turn the lights off")[0], "en")
        self.assertEqual(identifier.identify("выключи пожалуйста свет")[0], "ru")
        self.assertIsNone(identifier.identify("   ")[0])

    def test_an_unseen_language_still_returns_a_ranked_guess(self) -> None:
        identifier = gating.LanguageIdentifier(order=3).fit({"en": ["hello there"], "ru": ["привет"]})
        language, margin = identifier.identify("bonjour")
        self.assertIn(language, {"en", "ru"})
        self.assertGreaterEqual(margin, 0.0)


class ReplyShapeTests(unittest.TestCase):
    def test_turn_markers_are_not_content(self) -> None:
        self.assertEqual(gating.clean_reply("</s><s>Lights off.</s>"), "Lights off.")

    def test_a_parroted_request_is_an_echo_and_an_answer_is_not(self) -> None:
        self.assertTrue(gating.is_echo("turn off the lights please", "turn off the lights please"))
        self.assertFalse(gating.is_echo("Sure, switching them off now.", "turn off the lights please"))

    def test_a_repeating_tail_is_a_loop(self) -> None:
        self.assertTrue(gating.is_looping("tu as tu as tu as"))
        self.assertTrue(gating.is_looping("Schade bitte die Lichter aus Schade bitte die Lichter aus Schade bitte die Lichter aus"))
        self.assertFalse(gating.is_looping("Sure, I have switched the lights off for you."))

    def test_token_f1_is_one_for_identical_replies_and_zero_when_disjoint(self) -> None:
        self.assertEqual(gating.token_f1("lights off", "lights off"), 1.0)
        self.assertEqual(gating.token_f1("lights off", "weather sunny"), 0.0)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.identifier = gating.LanguageIdentifier(order=3).fit(
            {
                "en": ["turn off the lights please", "sure i have switched them off"],
                "ru": ["выключи свет пожалуйста", "конечно я выключил свет"],
            }
        )

    def _row(self, prompt_id: str, language: str, reply: str, identifier: str = "1") -> dict:
        return {
            "path": "text",
            "prompt_id": prompt_id,
            "language": language,
            "id": identifier,
            "intent": "iot_hue_lightoff",
            "utterance": "выключи свет пожалуйста",
            "reply": reply,
        }

    def test_a_cell_reports_obedience_and_usability_separately(self) -> None:
        rows = [
            self._row("B_input_language", "ru", "конечно я выключил свет", "1"),
            self._row("B_input_language", "ru", "Sure, I have switched them off", "2"),
            # right language, but a loop is not a target
            self._row("B_input_language", "ru", "свет свет свет свет", "3"),
            self._row("B_input_language", "ru", "", "4"),
        ]
        scored = gating.score_rows(rows, self.identifier)
        cell = scored["cells"]["text|B_input_language|ru"]
        self.assertEqual(cell["n"], 4)
        self.assertEqual(cell["answered_rate"], 0.75)
        self.assertEqual(cell["expected_output_language"], "ru")
        self.assertAlmostEqual(cell["expected_language_rate"], 0.5)
        self.assertAlmostEqual(cell["usable_target_rate"], 0.25)
        self.assertAlmostEqual(cell["looping_rate"], 0.25)

    def test_condition_a_is_graded_on_english_output(self) -> None:
        rows = [self._row("A_english_only", "ru", "Sure, I have switched them off", str(index))
                for index in range(4)]
        scored = gating.score_rows(rows, self.identifier)
        cell = scored["cells"]["text|A_english_only|ru"]
        self.assertEqual(cell["expected_output_language"], "en")
        self.assertAlmostEqual(cell["usable_target_rate"], 1.0)

    def test_a_verdict_fails_on_the_worst_foreign_language(self) -> None:
        cells = {
            "text|B_input_language|fr": {
                "path": "text", "prompt_id": "B_input_language",
                "input_language": "fr", "usable_target_rate": 0.95,
            },
            "text|B_input_language|ru": {
                "path": "text", "prompt_id": "B_input_language",
                "input_language": "ru", "usable_target_rate": 0.10,
            },
            "text|B_input_language|en": {
                "path": "text", "prompt_id": "B_input_language",
                "input_language": "en", "usable_target_rate": 1.0,
            },
        }
        verdicts = gating.gate_verdicts(cells)
        verdict = verdicts["text|B_input_language"]
        self.assertFalse(verdict["passes"])
        self.assertAlmostEqual(verdict["worst_language_rate"], 0.10)
        # the English control never decides a foreign-language condition
        self.assertNotIn("en", verdict["usable_target_rate_by_language"])


class TargetGapTests(unittest.TestCase):
    def test_the_gap_pairs_each_text_source_against_the_audio_target(self) -> None:
        identifier = gating.LanguageIdentifier(order=3).fit(
            {"en": ["the factorial of five is one hundred and twenty"], "ru": ["привет как дела"]}
        )
        rows = [
            {"id": "a", "source": "audio", "reply": "The factorial of five is 120."},
            {"id": "a", "source": "text_perception", "reply": "The factorial of five is 120."},
            {"id": "a", "source": "text_chat", "reply": "It equals one hundred and twenty."},
        ]
        gap = gating.score_target_gap(rows, identifier)
        self.assertEqual(gap["summary"]["text_perception"]["median_token_f1"], 1.0)
        self.assertLess(gap["summary"]["text_chat"]["median_token_f1"], 1.0)
        self.assertEqual(gap["summary"]["text_perception"]["n"], 1)

    def test_a_clip_without_an_audio_target_is_not_paired(self) -> None:
        identifier = gating.LanguageIdentifier(order=3).fit({"en": ["hello"], "ru": ["привет"]})
        gap = gating.score_target_gap(
            [{"id": "a", "source": "text_chat", "reply": "hello"}], identifier
        )
        self.assertEqual(gap["pairs"], [])


class PromptTests(unittest.TestCase):
    def test_both_framings_carry_the_utterance_and_an_unknown_path_is_rejected(self) -> None:
        system = gating.SYSTEM_PROMPTS["A_english_only"]
        for path in gating.PATHS:
            rendered = gating.render_prompt(path, system, "выключи свет")
            self.assertIn("выключи свет", rendered)
            self.assertIn(system, rendered)
        with self.assertRaises(ExperimentValidationError):
            gating.render_prompt("websocket", system, "hello")

    def test_the_frozen_prompts_stay_ascii_for_the_deployment_runtime(self) -> None:
        # render_system_prompt in the pinned bridge drops non-ASCII, so a
        # condition whose prompt is not ASCII cannot be served at all.
        for prompt in gating.SYSTEM_PROMPTS.values():
            self.assertTrue(prompt.isascii())


if __name__ == "__main__":
    unittest.main()
