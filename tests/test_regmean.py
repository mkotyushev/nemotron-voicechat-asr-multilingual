from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from asr_align import evaluation, manifests, regmean
from asr_align.encoder import Encoder
from asr_align.experiments import ExperimentValidationError, sha256_file

TINY_CONFIG = {
    "num_hidden_layers": 2,
    "hidden_size": 32,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "num_mel_bins": 16,
    "conv_kernel_size": 5,
    "sliding_window": 9,
}
TINY_CHANNELS = 8


def _hyper() -> regmean.Hyper:
    return regmean.hyper_from_config(TINY_CONFIG, TINY_CHANNELS)


def _state(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = Encoder(_hyper())
    return {f"encoder.{key}": value.detach().clone() for key, value in model.state_dict().items()}


def _mels(seed: int, batches: int = 3, batch: int = 4, frames: int = 400) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch, TINY_CONFIG["num_mel_bins"], frames, generator=generator)
        for _ in range(batches)
    ]


class RoutingTests(unittest.TestCase):
    def test_every_tensor_is_routed_exactly_once(self) -> None:
        state = _state(0)
        routing = regmean.route_tensors(state, _hyper(), regmean.MergePlan(alpha=0.5))
        self.assertEqual(routing["count"], len(state))
        self.assertEqual(
            routing["counts"]["regmean"] + routing["counts"]["average"], len(state)
        )
        self.assertEqual(set(routing["tensors"]), set(state))

    def test_pointwise_convolutions_are_on_the_regmean_path(self) -> None:
        routing = regmean.route_tensors(_state(0), _hyper(), regmean.MergePlan(alpha=0.5))
        for which in (1, 2):
            key = f"encoder.layers.0.conv.pointwise_conv{which}.weight"
            self.assertEqual(routing["tensors"][key]["route"], "regmean")
        self.assertEqual(
            routing["tensors"]["encoder.layers.0.conv.depthwise_conv.weight"]["route"],
            "average",
        )
        for key in (
            "encoder.layers.0.norm_out.weight",
            "encoder.layers.0.conv.norm.bias",
            "encoder.layers.0.self_attn.bias_u",
            "encoder.subsampling.conv_in.weight",
            "encoder.subsampling.linear.bias",
        ):
            self.assertEqual(routing["tensors"][key]["route"], "average", key)

    def test_a_declared_subset_moves_the_rest_to_averaging(self) -> None:
        plan = regmean.MergePlan(alpha=0.5, kinds=("ffn",), depths=(2,))
        routing = regmean.route_tensors(_state(0), _hyper(), plan)
        self.assertEqual(
            routing["tensors"]["encoder.layers.1.feed_forward1.linear1.weight"]["route"],
            "regmean",
        )
        for key in (
            "encoder.layers.0.feed_forward1.linear1.weight",
            "encoder.layers.1.self_attn.q_proj.weight",
        ):
            self.assertEqual(routing["tensors"][key]["route"], "average", key)

    def test_a_missing_linear_is_an_error(self) -> None:
        state = _state(0)
        del state["encoder.layers.0.self_attn.q_proj.weight"]
        with self.assertRaises(ExperimentValidationError):
            regmean.route_tensors(state, _hyper(), regmean.MergePlan(alpha=0.5))

    def test_the_widest_linear_input_is_the_subsampling_projection(self) -> None:
        # The deployed widths, not the toy's: 128 mel bins survive three
        # stride-2 convolutions as 17, and 256 channels x 17 = 4352 beats the
        # 4096-wide FFN. That is what sizes Dataset A, and getting it wrong
        # solves the widest layer at half the intended rows per dimension.
        deployed = regmean.hyper_from_config(
            {**TINY_CONFIG, "hidden_size": 1024, "intermediate_size": 4096,
             "num_mel_bins": 128, "num_attention_heads": 8, "num_hidden_layers": 24},
            256,
        )
        self.assertEqual(regmean.subsampling_input_dim(deployed), 4352)
        widest = max(regmean.linear_sites(deployed), key=lambda site: site.in_features)
        self.assertEqual((widest.module, widest.in_features), ("linear", 4352))
        self.assertEqual(
            regmean.GRAM_ROWS_PER_INPUT_DIMENSION * widest.in_features, 17408
        )


class PlanTests(unittest.TestCase):
    def test_alpha_one_is_rejected(self) -> None:
        with self.assertRaises(ExperimentValidationError):
            regmean.MergePlan(alpha=1.0)
        with self.assertRaises(ExperimentValidationError):
            regmean.MergePlan(alpha=0.0)

    def test_the_frozen_grid_excludes_alpha_one(self) -> None:
        self.assertNotIn(1.0, regmean.ALPHAS)
        for alpha in regmean.ALPHAS:
            regmean.MergePlan(alpha=alpha)

    def test_an_unknown_layernorm_source_is_rejected(self) -> None:
        with self.assertRaises(ExperimentValidationError):
            regmean.MergePlan(alpha=0.5, layernorm_source="PT_ML")


class GramTests(unittest.TestCase):
    def test_the_gram_is_normalized_by_its_rows(self) -> None:
        torch.manual_seed(0)
        rows = torch.randn(64, 8)
        one = regmean.GramAccumulator(8)
        one.update(rows)
        doubled = regmean.GramAccumulator(8)
        doubled.update(rows)
        doubled.update(rows)
        self.assertEqual(doubled.rows, 2 * one.rows)
        torch.testing.assert_close(doubled.gram(), one.gram(), rtol=1e-9, atol=1e-9)

    def test_clip_length_cannot_act_as_a_merge_coefficient(self) -> None:
        torch.manual_seed(1)
        short = torch.randn(16, 8)
        accumulator = regmean.GramAccumulator(8)
        accumulator.update(short)
        expected = (short.double().T @ short.double()) / 16
        torch.testing.assert_close(accumulator.gram(), expected, rtol=1e-6, atol=1e-8)

    def test_an_empty_accumulator_is_an_error(self) -> None:
        with self.assertRaises(ExperimentValidationError):
            regmean.GramAccumulator(4).gram()

    def test_the_width_must_match(self) -> None:
        with self.assertRaises(ExperimentValidationError):
            regmean.GramAccumulator(4).update(torch.zeros(3, 5))


class SolveTests(unittest.TestCase):
    @staticmethod
    def _grams(seed: int, width: int, rows: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(rows, width, generator=generator, dtype=torch.float64)
        return (x.T @ x) / rows

    def test_equal_grams_reduce_to_the_unweighted_mean(self) -> None:
        gram = self._grams(0, 6, 256)
        torch.manual_seed(0)
        weights = {"F": torch.randn(4, 6, dtype=torch.float64), "M": torch.randn(4, 6, dtype=torch.float64)}
        merged, diagnostics = regmean.regmean_solve({"F": gram, "M": gram}, weights, 0.5)
        expected = (weights["F"] + weights["M"]) / 2
        torch.testing.assert_close(merged.double(), expected, rtol=1e-6, atol=1e-6)
        self.assertTrue(diagnostics["reduces_to_mean"])

    def test_different_grams_do_not_reduce_to_the_mean(self) -> None:
        torch.manual_seed(0)
        weights = {"F": torch.randn(4, 6, dtype=torch.float64), "M": torch.randn(4, 6, dtype=torch.float64)}
        merged, diagnostics = regmean.regmean_solve(
            {"F": self._grams(0, 6, 256), "M": self._grams(1, 6, 256) * 4.0}, weights, 0.9
        )
        self.assertFalse(diagnostics["reduces_to_mean"])
        self.assertGreater(diagnostics["distance_from_mean"], 1e-3)
        self.assertEqual(sorted(diagnostics["output_residual_relative"]), ["F", "M"])

    def test_eq_2_matches_a_brute_force_least_squares(self) -> None:
        alpha = 0.7
        grams = {"F": self._grams(2, 5, 128), "M": self._grams(3, 5, 128)}
        torch.manual_seed(2)
        weights = {"F": torch.randn(3, 5, dtype=torch.float64), "M": torch.randn(3, 5, dtype=torch.float64)}
        merged, _ = regmean.regmean_solve(grams, weights, alpha)
        shrunk = {name: regmean.shrink(grams[name], alpha) for name in grams}
        left = sum(shrunk.values())
        right = sum(shrunk[name] @ weights[name].T for name in grams)
        expected = torch.linalg.lstsq(left, right).solution.T
        torch.testing.assert_close(merged.double(), expected, rtol=1e-6, atol=1e-6)

    def test_a_near_null_direction_cannot_blow_the_weights_up(self) -> None:
        # The unridged solve turned round-off in the Gram's dead directions into
        # encoder weights 8e14 times PT_ML's norm on the real checkpoints. The
        # solve may only move where the data determines it.
        # Four channels that never fire and two that are near-duplicates: the
        # shape a post-ReLU convolution actually produces, and the one the
        # (1 - alpha) diag(G) shrinkage cannot rescue, because it is zero there
        # too.
        width, rows = 24, 512
        generator = torch.Generator().manual_seed(9)

        def gram(seed: int, scale: float) -> torch.Tensor:
            x = torch.randn(rows, width, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
            x[:, 20:] = 0.0
            x[:, 1] = x[:, 0] + 1e-9 * x[:, 1]
            x *= scale
            return (x.T @ x) / rows

        gram_f, gram_m = gram(1, 1.0), gram(2, 1.7)
        weights = {
            "F": torch.randn(8, width, generator=generator, dtype=torch.float64),
            "M": torch.randn(8, width, generator=generator, dtype=torch.float64),
        }
        merged, diagnostics = regmean.regmean_solve({"F": gram_f, "M": gram_m}, weights, 0.9)
        largest = max(float(value.norm()) for value in weights.values())
        self.assertLess(float(merged.double().norm()), 4.0 * largest)
        self.assertLess(diagnostics["effective_rank"], width)

    def test_shrinkage_interpolates_between_the_gram_and_its_diagonal(self) -> None:
        gram = self._grams(4, 5, 128)
        torch.testing.assert_close(regmean.shrink(gram, 1.0), gram)
        torch.testing.assert_close(
            regmean.shrink(gram, 0.0), torch.diag(torch.diagonal(gram))
        )

    def test_a_dead_input_dimension_keeps_the_candidates_mean(self) -> None:
        # Eq. 2 says nothing about a direction the data never visits, so the
        # ridge decides it. Toward zero it would delete those weights.
        gram = torch.zeros(4, 4, dtype=torch.float64)
        gram[:3, :3] = self._grams(5, 3, 64)
        torch.manual_seed(5)
        weights = {"F": torch.randn(2, 4, dtype=torch.float64), "M": torch.randn(2, 4, dtype=torch.float64)}
        merged, diagnostics = regmean.regmean_solve({"F": gram, "M": gram * 2.0}, weights, 0.5)
        self.assertGreater(diagnostics["ridge"], 0.0)
        self.assertEqual(diagnostics["ridge_target"], "the candidates' mean")
        self.assertLess(diagnostics["effective_rank"], diagnostics["input_dimension"])
        expected = (weights["F"][:, 3] + weights["M"][:, 3]) / 2
        torch.testing.assert_close(merged.double()[:, 3], expected, rtol=1e-6, atol=1e-6)


class AverageTests(unittest.TestCase):
    def test_the_average_is_elementwise(self) -> None:
        left, right = _state(0), _state(1)
        averaged = regmean.simple_average({"M": left, "F": right})
        self.assertEqual(set(averaged), set(left))
        key = "encoder.layers.0.norm_out.weight"
        torch.testing.assert_close(averaged[key], (left[key] + right[key]) / 2)

    def test_mismatched_keys_are_rejected(self) -> None:
        left, right = _state(0), _state(1)
        del right["encoder.layers.0.norm_out.bias"]
        with self.assertRaises(ExperimentValidationError):
            regmean.simple_average({"M": left, "F": right})


class MergeTests(unittest.TestCase):
    def test_the_merge_covers_every_tensor_and_keeps_every_shape(self) -> None:
        states = {"M": _state(0), "F": _state(1)}
        merged, report = regmean.merge_encoders(
            states, TINY_CONFIG, {"M": _mels(10), "F": _mels(11)},
            regmean.MergePlan(alpha=0.5),
        )
        self.assertEqual(set(merged), set(states["M"]))
        for key, value in merged.items():
            self.assertEqual(value.shape, states["M"][key].shape)
            self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(report["gram"]["frames_equalized_across_candidates"])
        self.assertTrue(report["gram"]["normalized_by_frame_count"])

    def test_the_position_projection_collapses_onto_the_mean(self) -> None:
        # `relative_k_proj` reads the position encoding, which does not depend on
        # the data, so its two Grams coincide whatever audio each candidate saw.
        states = {"M": _state(0), "F": _state(1)}
        merged, report = regmean.merge_encoders(
            states, TINY_CONFIG, {"M": _mels(10), "F": _mels(11)},
            regmean.MergePlan(alpha=0.5),
        )
        key = "encoder.layers.0.self_attn.relative_k_proj.weight"
        self.assertIn(key, report["degenerate_solves"])
        torch.testing.assert_close(
            merged[key], (states["M"][key] + states["F"][key]) / 2, rtol=1e-5, atol=1e-5
        )

    def test_the_left_context_comes_from_the_supplied_configuration(self) -> None:
        states = {"M": _state(0), "F": _state(1)}
        _, report = regmean.merge_encoders(
            states, TINY_CONFIG, {"M": _mels(10), "F": _mels(11)},
            regmean.MergePlan(alpha=0.5),
        )
        self.assertEqual(
            report["runtime_configuration"]["attention_left_context"],
            TINY_CONFIG["sliding_window"] - 1,
        )
        self.assertTrue(report["runtime_configuration"]["ft_en_gram_collected_at_unseen_context"])

    def test_seeding_the_layernorms_from_ft_en_takes_ft_en_exactly(self) -> None:
        states = {"M": _state(0), "F": _state(1)}
        merged, _ = regmean.merge_encoders(
            states, TINY_CONFIG, {"M": _mels(10), "F": _mels(11)},
            regmean.MergePlan(alpha=0.5, layernorm_source="F"),
        )
        for key in (
            "encoder.layers.0.norm_out.weight",
            "encoder.layers.1.norm_conv.bias",
            "encoder.layers.0.conv.norm.weight",
        ):
            torch.testing.assert_close(merged[key], states["F"][key])

    def test_plain_regmean_and_the_cross_layer_variant_differ(self) -> None:
        states = {"M": _state(0), "F": _state(1)}
        mels = {"M": _mels(10), "F": _mels(11)}
        plus, _ = regmean.merge_encoders(states, TINY_CONFIG, mels, regmean.MergePlan(alpha=0.5))
        plain, _ = regmean.merge_encoders(
            states, TINY_CONFIG, mels, regmean.MergePlan(alpha=0.5, cross_layer=False)
        )
        key = "encoder.layers.1.feed_forward2.linear2.weight"
        self.assertGreater(float((plus[key] - plain[key]).abs().max()), 0.0)

    def test_a_third_candidate_is_rejected(self) -> None:
        states = {"M": _state(0), "F": _state(1), "E": _state(2)}
        with self.assertRaises(ExperimentValidationError):
            regmean.merge_encoders(
                states, TINY_CONFIG,
                {"M": _mels(10), "F": _mels(11), "E": _mels(12)},
                regmean.MergePlan(alpha=0.5),
            )

    def test_merging_a_checkpoint_with_itself_returns_it(self) -> None:
        state = _state(3)
        merged, _ = regmean.merge_encoders(
            {"M": state, "F": {key: value.clone() for key, value in state.items()}},
            TINY_CONFIG, {"M": _mels(10), "F": _mels(11)}, regmean.MergePlan(alpha=0.5),
        )
        for key, value in merged.items():
            torch.testing.assert_close(value, state[key], rtol=2e-4, atol=2e-4, msg=key)


class AgreementTests(unittest.TestCase):
    def test_a_model_agrees_perfectly_with_itself(self) -> None:
        state = _state(0)
        scores = regmean.output_agreement(state, state, TINY_CONFIG, _mels(20))
        self.assertAlmostEqual(scores["r2"], 1.0, places=9)
        self.assertAlmostEqual(scores["cosine_mean"], 1.0, places=6)
        self.assertEqual(scores["relative_error"], 0.0)

    def test_the_selection_score_averages_both_candidates(self) -> None:
        score = regmean.selection_score({"M": {"r2": 0.2}, "F": {"r2": 0.6}})
        self.assertAlmostEqual(score, 0.4)

    def test_the_selection_score_needs_both_candidates(self) -> None:
        with self.assertRaises(ExperimentValidationError):
            regmean.selection_score({"M": {"r2": 0.2}})


class DatasetAManifestTests(unittest.TestCase):
    @staticmethod
    def _clip(root: Path, name: str) -> dict[str, object]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode("utf-8"))
        return {
            "path": name,
            "offset": 0,
            "n_samples": 48000,
            "source_frames": 48000,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def _slurp(self, root: Path) -> dict[str, object]:
        splits = {
            "gram": [{**self._clip(root, "slurp/a.flac"), "utterance_id": "1", "slurp_id": 1,
                      "recording": "a.flac", "intent": "alarm_set", "scenario": "alarm"}],
            "heldout": [{**self._clip(root, "slurp/b.flac"), "utterance_id": "2", "slurp_id": 2,
                         "recording": "b.flac", "intent": "alarm_set", "scenario": "alarm"}],
        }
        return manifests.build_slurp_manifest(
            root, splits=splits, source={"corpus": "SLURP"}, selection={"order": "test"},
            seconds=3.0,
        )

    def _speech_massive(self, root: Path, languages=("fr-FR", "de-DE")) -> dict[str, object]:
        splits = {"gram": [], "heldout": []}
        for index, language in enumerate(languages):
            for split in splits:
                name = f"speech-massive/{language}/{split}{index}.flac"
                splits[split].append(
                    {**self._clip(root, name), "utterance_id": f"{language}:{split}",
                     "language": language, "intent": "alarm_set", "speaker_id": "s"}
                )
        return manifests.build_speech_massive_manifest(
            root, splits=splits, source={"corpus": "Speech-MASSIVE"},
            selection={"order": "test"}, languages=list(languages), seconds=3.0,
        )

    def test_both_manifests_validate_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for payload in (self._slurp(root), self._speech_massive(root)):
                manifests.validate_dataset_a_manifest(payload)
                manifests.verify_audio_files(payload, root=root)

    def test_a_changed_recording_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._slurp(root)
            (root / "slurp/a.flac").write_bytes(b"different")
            with self.assertRaises(ExperimentValidationError):
                manifests.verify_audio_files(payload, root=root)

    def test_a_clip_may_not_appear_in_both_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._slurp(root)
            payload["splits"]["heldout"] = list(payload["splits"]["gram"])
            with self.assertRaises(ExperimentValidationError):
                manifests.validate_slurp_manifest(manifests._with_digest(payload))

    def test_the_multilingual_gram_may_not_drop_a_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._speech_massive(root)
            payload["splits"]["gram"] = payload["splits"]["gram"][:1]
            with self.assertRaises(ExperimentValidationError):
                manifests.validate_speech_massive_manifest(manifests._with_digest(payload))

    def test_a_frozen_manifest_is_written_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self._slurp(root)
            path = root / "slurp.json"
            manifests.write_frozen(path, payload)
            manifests.write_frozen(path, payload)
            changed = dict(payload)
            changed["seed"] = 7
            with self.assertRaises(ExperimentValidationError):
                manifests.write_frozen(path, changed)

    def test_shrinkage_may_only_be_selected_on_a_held_out_dataset_a_split(self) -> None:
        manifests.assert_merge_selection_source("SLURP", "heldout")
        manifests.assert_merge_selection_source("Speech-MASSIVE", "heldout")
        for dataset, split in (
            ("FLEURS", "heldout"),
            ("LibriSpeech", "validation"),
            ("SLURP", "gram"),
        ):
            with self.assertRaises(ExperimentValidationError):
                manifests.assert_merge_selection_source(dataset, split)


class ResultContractTests(unittest.TestCase):
    @staticmethod
    def _inputs(seed: int = 0) -> dict[str, object]:
        rng = np.random.default_rng(seed)
        english = rng.normal(size=(32, 6))
        probe = rng.normal(size=(12, 6))
        return {
            "english_prediction": english + 0.1 * rng.normal(size=english.shape),
            "english_target": english,
            "pt_ml_english_prediction": english + 0.2 * rng.normal(size=english.shape),
            "retrieval_inputs": {
                task: {"fr_fr": (probe, probe + 0.1, probe + 0.2, probe + 0.3)}
                for task in evaluation.RETRIEVAL_TASKS
            },
            "diagnostic_embeddings": {"probe": probe},
            "manifest_hashes": {"librispeech": "a", "fleurs": "b"},
        }

    def test_comparison_6_records_a_null_lambda(self) -> None:
        result = evaluation.evaluate_candidate(
            comparison=6, candidate_id="regmean-plus-plus", weight=None,
            precision="pre_quantization", **self._inputs(),
        )
        self.assertIsNone(result["lambda"])
        self.assertEqual(result["schema_version"], evaluation.RESULT_SCHEMA_VERSION)
        evaluation.validate_result(result)

    def test_comparison_6_rejects_a_lambda(self) -> None:
        with self.assertRaises(ExperimentValidationError):
            evaluation.evaluate_candidate(
                comparison=6, candidate_id="regmean-plus-plus", weight=1.0,
                precision="pre_quantization", **self._inputs(),
            )

    def test_the_lambda_comparisons_still_require_one(self) -> None:
        with self.assertRaises(ExperimentValidationError):
            evaluation.evaluate_candidate(
                comparison=2, candidate_id="direct-lambda-1", weight=None,
                precision="pre_quantization", **self._inputs(),
            )

    def test_frozen_schema_1_0_records_remain_valid(self) -> None:
        result = evaluation.evaluate_candidate(
            comparison=1, candidate_id="pt-ml-baseline", weight=0.0,
            precision="pre_quantization", **self._inputs(),
        )
        result["schema_version"] = "1.0"
        result["lambda"] = 0.0
        evaluation.validate_result(result)

    def test_a_precision_pair_may_not_straddle_schema_versions(self) -> None:
        pre = evaluation.evaluate_candidate(
            comparison=6, candidate_id="regmean-plus-plus", weight=None,
            precision="pre_quantization", **self._inputs(),
        )
        post = evaluation.evaluate_candidate(
            comparison=6, candidate_id="regmean-plus-plus", weight=None,
            precision="post_quantization", **self._inputs(),
        )
        evaluation.validate_precision_pair(pre, post)
        post["schema_version"] = "1.0"
        with self.assertRaises(ExperimentValidationError):
            evaluation.validate_precision_pair(pre, post)


class DeltaTableTests(unittest.TestCase):
    def _result(self, comparison: int, weight: float | None, seed: int) -> dict[str, object]:
        return evaluation.evaluate_candidate(
            comparison=comparison,
            candidate_id=f"c{comparison}",
            weight=weight,
            precision="pre_quantization",
            **ResultContractTests._inputs(seed),
        )

    def test_the_table_reports_both_differences_against_pt_ml(self) -> None:
        table = regmean.delta_table(
            self._result(6, None, 0), self._result(1, 0.0, 1), self._result(2, 1.0, 2)
        )
        english = table["english_voicechat_space"]["r2"]
        self.assertIn("paired_interval", english["merge"])
        self.assertIn("comparison_2_lambda_1", english)
        self.assertAlmostEqual(
            english["merge"]["difference_vs_pt_ml"],
            english["merge"]["value"] - english["pt_ml"],
        )

    def test_the_table_refuses_a_mismatched_comparison(self) -> None:
        with self.assertRaises(ExperimentValidationError):
            regmean.delta_table(self._result(6, None, 0), self._result(2, 1.0, 1))


if __name__ == "__main__":
    unittest.main()
