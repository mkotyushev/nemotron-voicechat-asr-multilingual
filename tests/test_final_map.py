from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from asr_align import evaluation, export, final_map
from asr_align.experiments import ExperimentValidationError, sha256_file
from asr_align.interface import AffineMap, Moments


def _paired_moments(
    source: torch.Tensor, target: torch.Tensor
) -> Moments:
    moments = Moments(source.shape[1], target.shape[1])
    moments.update(source, target)
    return moments


def _linear_pair(
    width: int = 4, frames: int = 512, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.eye(width, dtype=torch.float64) + 0.25 * torch.randn(
        width, width, generator=generator, dtype=torch.float64
    )
    bias = 0.1 * torch.randn(width, generator=generator, dtype=torch.float64)
    x = torch.randn(frames, width, generator=generator, dtype=torch.float64)
    y = x @ weight + bias
    return x.float(), y.float(), weight, bias


class IdentityRidgeTests(unittest.TestCase):
    def test_weak_penalty_recovers_the_true_map(self) -> None:
        x, y, weight, bias = _linear_pair()
        fitted = final_map.identity_ridge_map(_paired_moments(x, y), 1e-8, "A_L")
        self.assertLess(float((fitted.weight - weight).abs().max()), 1e-4)
        self.assertLess(float((fitted.bias - bias).abs().max()), 1e-4)
        scored = final_map.score_map(fitted, x, y)
        self.assertGreater(scored["r2"], 0.999)
        self.assertEqual(scored["n_frames"], x.shape[0])

    def test_strong_penalty_degenerates_to_the_untouched_interface(self) -> None:
        x, y, _, _ = _linear_pair()
        fitted = final_map.identity_ridge_map(_paired_moments(x, y), 1e6, "A_L")
        distance = final_map.identity_distance(fitted)
        self.assertLess(distance["relative_frobenius"], 1e-3)
        self.assertEqual(fitted.detail["regularized_toward"], "identity")
        self.assertEqual(fitted.detail["centering"], "training-set means")

    def test_invalid_penalties_and_shapes_are_rejected(self) -> None:
        x, y, _, _ = _linear_pair()
        with self.assertRaisesRegex(ExperimentValidationError, "alpha"):
            final_map.identity_ridge_map(_paired_moments(x, y), -1.0, "A_L")
        rectangular = Moments(4, 3)
        rectangular.update(torch.randn(8, 4), torch.randn(8, 3))
        with self.assertRaisesRegex(ExperimentValidationError, "square"):
            final_map.identity_ridge_map(rectangular, 1.0, "A_L")

    def test_selection_prefers_the_best_held_out_fit_and_keeps_the_sweep(self) -> None:
        x, y, _, _ = _linear_pair(frames=1024)
        held_x, held_y = x[:256], y[:256]
        selected, trace = final_map.select_alpha(
            _paired_moments(x[256:], y[256:]),
            held_x,
            held_y,
            name="A_L",
            alphas=(1e-6, 1.0, 100.0),
        )
        self.assertEqual(len(trace), 3)
        self.assertEqual(
            selected.detail["alpha"], max(trace, key=lambda row: row["r2"])["alpha"]
        )
        # The true relationship is linear, so the weakest penalty must win.
        self.assertEqual(selected.detail["alpha"], 1e-6)
        for row in trace:
            self.assertIn("condition_number", row)
            self.assertIn("identity_relative_frobenius", row)


class MapDiagnosticsTests(unittest.TestCase):
    def test_conditioning_and_identity_distance_of_a_known_map(self) -> None:
        weight = torch.diag(torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64))
        mapping = AffineMap(weight, torch.zeros(3, dtype=torch.float64), "diag")
        report = final_map.conditioning(mapping)
        self.assertAlmostEqual(report["condition_number"], 4.0, places=6)
        self.assertEqual(report["singular_values"]["count"], 3)
        distance = final_map.identity_distance(mapping)
        self.assertAlmostEqual(distance["off_diagonal_frobenius"], 0.0, places=12)
        self.assertAlmostEqual(distance["max_abs"], 3.0, places=12)

    def test_cycle_consistency_is_exact_for_a_map_and_its_inverse(self) -> None:
        x, y, _, _ = _linear_pair(frames=1024)
        forward = final_map.identity_ridge_map(_paired_moments(x, y), 1e-8, "A_L")
        reverse = final_map.identity_ridge_map(_paired_moments(y, x), 1e-8, "B_L")
        cycle = final_map.cycle_consistency(forward, reverse)
        for row in cycle["round_trips"].values():
            self.assertLess(row["relative_frobenius"], 1e-3)
        composed = final_map.compose(forward, reverse, "A_L B_L")
        self.assertGreater(final_map.score_map(composed, x, x)["r2"], 0.999)


class FoldingTests(unittest.TestCase):
    def test_folding_reproduces_map_then_project(self) -> None:
        generator = torch.Generator().manual_seed(1)
        proj_weight = torch.randn(7, 4, generator=generator)
        proj_bias = torch.randn(7, generator=generator)
        _, _, weight, bias = _linear_pair(width=4)
        mapping = AffineMap(weight, bias, "B_L")
        folded_weight, folded_bias, report = final_map.fold_reverse_map(
            mapping, proj_weight, proj_bias
        )
        self.assertEqual(tuple(folded_weight.shape), (7, 4))
        self.assertEqual(tuple(folded_bias.shape), (7,))
        self.assertEqual(report["formula"]["weight"], "W_proj,M = W_proj,F B_L^T")
        hidden = torch.randn(16, 4, generator=generator)
        check = final_map.verify_folding(
            mapping, proj_weight, proj_bias, folded_weight, folded_bias, hidden
        )
        self.assertTrue(check["passed"])
        self.assertLess(check["relative_l2"], final_map.FOLD_RELATIVE_TOLERANCE)

        with self.assertRaisesRegex(ExperimentValidationError, "does not reproduce"):
            final_map.verify_folding(
                mapping,
                proj_weight,
                proj_bias,
                folded_weight + 1.0,
                folded_bias,
                hidden,
            )

    def test_a_map_of_the_wrong_width_is_rejected(self) -> None:
        mapping = AffineMap(
            torch.eye(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), "B_L"
        )
        with self.assertRaisesRegex(ExperimentValidationError, "encoder output"):
            final_map.fold_reverse_map(mapping, torch.randn(7, 5), torch.randn(7))


class EncoderByteIdentityTests(unittest.TestCase):
    def test_signed_zero_is_not_byte_identical(self) -> None:
        state = {"encoder.subsampling.linear.bias": torch.tensor([0.0, 1.0])}
        same = {"encoder.subsampling.linear.bias": torch.tensor([0.0, 1.0])}
        signed = {"encoder.subsampling.linear.bias": torch.tensor([-0.0, 1.0])}
        record = final_map.assert_encoder_byte_identical(state, same, label="clone")
        self.assertTrue(record["byte_identical"])
        self.assertEqual(record["tensor_count"], 1)
        # torch.equal calls these equal; the artifact claim is stronger than that.
        self.assertTrue(torch.equal(state["encoder.subsampling.linear.bias"], signed["encoder.subsampling.linear.bias"]))
        with self.assertRaisesRegex(ExperimentValidationError, "byte-identical"):
            final_map.assert_encoder_byte_identical(state, signed, label="signed zero")

    def test_non_encoder_tensors_are_not_hashed(self) -> None:
        with self.assertRaisesRegex(ExperimentValidationError, "no canonical encoder"):
            final_map.encoder_byte_digest({"proj.weight": torch.zeros(2, 2)})
        digest = final_map.encoder_byte_digest(
            {"encoder.a": torch.zeros(2), "proj.weight": torch.zeros(2, 2)}
        )
        self.assertEqual(digest["tensor_count"], 1)


class ActivationCacheTests(unittest.TestCase):
    @staticmethod
    def _write_cache(root: Path, *, n_layer: int = 2, manifest_sha256: str = "abc") -> Path:
        activations = root / "activations" / "pt_ml-baseline"
        index = {
            "schema_version": "1.0",
            "comparison": 1,
            "candidate_id": "pt_ml-baseline",
            "precision": "pre_quantization",
            "manifest_sha256": manifest_sha256,
            "reserved_test_encoded": False,
            "hook_points": "subsampling output and every block output",
            "splits": {},
        }
        for split in ("map_train", "validation"):
            shard = activations / split / "batch-00000.safetensors"
            arrays = {"residual.subsampling": np.zeros((2, 3, 4), dtype=np.float32)}
            for block in range(n_layer):
                arrays[f"residual.block.{block}"] = np.full(
                    (2, 3, 4), float(block + 1), dtype=np.float32
                )
            export.write_safetensors(shard, arrays, {"split": split})
            index["splits"][split] = [
                {
                    "path": shard.relative_to(activations.parent).as_posix(),
                    "bytes": shard.stat().st_size,
                    "sha256": sha256_file(shard),
                    "split": split,
                    "records": [f"{split}/a.flac", f"{split}/b.flac"],
                    "tensors": {name: list(value.shape) for name, value in arrays.items()},
                }
            ]
        path = activations / "index.json"
        path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        return path

    def test_final_layer_is_located_and_content_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_cache(Path(directory))
            cache = final_map.load_activation_cache(
                path, manifest_sha256="abc", n_layer=2, candidate_id="pt_ml-baseline"
            )
            self.assertEqual(cache.final_key, "residual.block.1")
            shard = cache.records("map_train")[0]
            records, value = final_map.read_final_activations(cache, shard)
            self.assertEqual(records, ["map_train/a.flac", "map_train/b.flac"])
            self.assertEqual(list(value.shape), [2, 3, 4])
            self.assertTrue(bool((value == 2.0).all()))

            with self.assertRaisesRegex(ExperimentValidationError, "manifest"):
                final_map.load_activation_cache(
                    path, manifest_sha256="other", n_layer=2, candidate_id="pt_ml-baseline"
                )
            with self.assertRaisesRegex(ExperimentValidationError, "block outputs"):
                final_map.load_activation_cache(
                    path, manifest_sha256="abc", n_layer=3, candidate_id="pt_ml-baseline"
                )

    def test_a_tampered_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_cache(Path(directory))
            cache = final_map.load_activation_cache(
                path, manifest_sha256="abc", n_layer=2, candidate_id="pt_ml-baseline"
            )
            shard = cache.records("validation")[0]
            target = cache.root / shard["path"]
            # the trailing tensor is all zeros, so the tamper has to be a value change
            target.write_bytes(target.read_bytes()[:-4] + b"\xff\xff\x7f\x3f")
            with self.assertRaisesRegex(ExperimentValidationError, "SHA-256"):
                final_map.read_final_activations(cache, shard)


def _result(comparison: int, *, seed: int = 0, shift: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    target = rng.normal(size=(16, 4))
    prediction = target + shift + 0.1 * rng.normal(size=(16, 4))
    probe = target + 0.2 * rng.normal(size=(16, 4))
    retrieval = {
        task: {"fr_fr": (probe, target, probe + shift, target)}
        for task in evaluation.RETRIEVAL_TASKS
    }
    return evaluation.evaluate_candidate(
        comparison=comparison,
        candidate_id=final_map.CANDIDATE_ID if comparison == 3 else "pt_ml-baseline",
        weight=0.0,
        precision="pre_quantization",
        english_prediction=prediction,
        english_target=target,
        pt_ml_english_prediction=prediction + shift,
        retrieval_inputs=retrieval,
        diagnostic_embeddings={"english": prediction},
        manifest_hashes={"librispeech": "l", "fleurs": "f"},
        seed=0,
    )


class ComparisonDeltaTests(unittest.TestCase):
    def test_delta_table_carries_paired_intervals_from_the_candidate(self) -> None:
        candidate = _result(3, shift=0.05)
        reference = _result(1)
        table = final_map.comparison_delta_table(candidate, reference)
        self.assertEqual(table["direction"], "Comparison 3 minus Comparison 1")
        english = table["english_voicechat_space"]["r2"]
        self.assertAlmostEqual(
            english["difference"], english["final_map"] - english["pt_ml"], places=12
        )
        self.assertEqual(
            english["paired_interval"],
            candidate["evaluations"]["english_voicechat_space"]["confidence_intervals"][
                "difference_vs_pt_ml"
            ]["r2"],
        )
        for task in evaluation.RETRIEVAL_TASKS:
            row = table["retrieval"][task]["fr_fr"]
            self.assertIn("paired_interval", row["mrr"])
            self.assertIn("hit_count", row)

    def test_incomparable_rows_are_refused(self) -> None:
        candidate = _result(3)
        reference = _result(1)
        with self.assertRaisesRegex(ExperimentValidationError, "Comparison 3"):
            final_map.comparison_delta_table(reference, reference)
        mismatched = json.loads(json.dumps(reference))
        mismatched["manifests"]["fleurs"] = "other"
        with self.assertRaisesRegex(ExperimentValidationError, "manifests"):
            final_map.comparison_delta_table(candidate, mismatched)
        staged = json.loads(json.dumps(reference))
        staged["precision"] = "post_quantization"
        with self.assertRaisesRegex(ExperimentValidationError, "precision"):
            final_map.comparison_delta_table(candidate, staged)


if __name__ == "__main__":
    unittest.main()
