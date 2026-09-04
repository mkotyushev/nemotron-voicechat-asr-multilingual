from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from asr_align import evaluation, manifests
from asr_align.experiments import (
    LAMBDAS,
    PRIMARY_LAMBDA,
    EXPECTED_REPO_IDS,
    CheckpointIdentity,
    ExperimentValidationError,
    arithmetic_summary,
    assert_runtime_config_inherited,
    candidate,
    inherit_runtime_config,
    stable_json_sha256,
    task_vector,
    validate_encoder_triplet,
    verify_declared_ancestor,
)


def _state(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "encoder.a.weight": torch.tensor([[1.0 + offset, 2.0], [3.0, 4.0]]),
        "encoder.a.bias": torch.tensor([offset, 1.0 + offset]),
        "proj.weight": torch.full((2, 2), 99.0),
    }


class ArithmeticTests(unittest.TestCase):
    def test_frozen_sweep_and_primary(self) -> None:
        self.assertEqual(LAMBDAS, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(PRIMARY_LAMBDA, 1.0)

    def test_encoder_only_f32_arithmetic(self) -> None:
        e = _state(0.0)
        m = _state(10.0)
        f = _state(2.0)
        delta = task_vector(e, f)
        result = candidate(m, delta, 1.0)
        self.assertEqual(set(result), {"encoder.a.weight", "encoder.a.bias"})
        self.assertTrue(all(value.dtype == torch.float32 for value in result.values()))
        self.assertTrue(torch.equal(candidate(m, delta, 0.0)["encoder.a.bias"], m["encoder.a.bias"]))
        summary = arithmetic_summary(e, m, f)
        self.assertTrue(summary["lambda_zero_exact_pt_ml"])
        self.assertEqual(summary["primary_lambda"], 1.0)

    def test_rejects_missing_shape_nonfinite_and_unknown_lambda(self) -> None:
        e, m, f = _state(), _state(1), _state(2)
        bad_missing = dict(m)
        bad_missing.pop("encoder.a.bias")
        with self.assertRaises(ExperimentValidationError):
            validate_encoder_triplet(e, bad_missing, f)
        bad_shape = dict(m)
        bad_shape["encoder.a.bias"] = torch.ones(1, 2)
        with self.assertRaisesRegex(ExperimentValidationError, "broadcasting"):
            validate_encoder_triplet(e, bad_shape, f)
        bad_finite = dict(f)
        bad_finite["encoder.a.bias"] = torch.tensor([float("nan"), 1.0])
        with self.assertRaisesRegex(ExperimentValidationError, "NaN"):
            task_vector(e, bad_finite)
        with self.assertRaises(ExperimentValidationError):
            candidate(m, task_vector(e, f), 0.3)

    def test_runtime_config_is_exact_copy(self) -> None:
        config = {"encoder_config": {"sliding_window": 57}, "torch_dtype": "float32"}
        inherited = inherit_runtime_config(config)
        self.assertEqual(inherited, config)
        self.assertIsNot(inherited, config)
        inherited["encoder_config"]["sliding_window"] = 71
        with self.assertRaises(ExperimentValidationError):
            assert_runtime_config_inherited(inherited, config)

    def test_lineage_must_match_exact_pin_and_have_evidence(self) -> None:
        e = CheckpointIdentity("E", EXPECTED_REPO_IDS["E"], "abc1234", Path("en"), "asr")
        f = CheckpointIdentity(
            "F", EXPECTED_REPO_IDS["F"], "def5678", Path("ft"), "voicechat_container"
        )
        result = verify_declared_ancestor(
            e,
            f,
            {
                "repo_id": EXPECTED_REPO_IDS["E"],
                "revision": "abc1234",
                "evidence": "https://example.test/release-note",
            },
        )
        self.assertTrue(result["verified"])
        with self.assertRaises(ExperimentValidationError):
            verify_declared_ancestor(
                e,
                f,
                {
                    "repo_id": EXPECTED_REPO_IDS["E"],
                    "revision": "different",
                    "evidence": "https://example.test/release-note",
                },
            )


class ManifestTests(unittest.TestCase):
    def _libri(self) -> dict:
        splits = {}
        for index, split in enumerate(manifests.LIBRISPEECH_SPLITS):
            splits[split] = [{
                "path": f"{index}/1/{index}.flac",
                "speaker": str(index),
                "chapter": "1",
                "offset": 0,
                "n_samples": 16000,
                "bytes": 10,
                "sha256": "0" * 64,
            }]
        base = {
            "schema_version": manifests.MANIFEST_SCHEMA_VERSION,
            "dataset": "LibriSpeech",
            "root": "/data",
            "splits": splits,
        }
        base["manifest_sha256"] = stable_json_sha256(base)
        return base

    def test_manifest_digest_disjointness_and_freeze(self) -> None:
        payload = self._libri()
        manifests.validate_librispeech_manifest(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            first = manifests.write_frozen(path, payload)
            second = manifests.write_frozen(path, payload)
            self.assertEqual(first, second)
            changed = copy.deepcopy(payload)
            changed["splits"]["test"][0]["path"] = "2/1/other.flac"
            changed["manifest_sha256"] = stable_json_sha256(
                {key: value for key, value in changed.items() if key != "manifest_sha256"}
            )
            with self.assertRaisesRegex(ExperimentValidationError, "refusing"):
                manifests.write_frozen(path, changed)

    def test_speaker_leakage_is_rejected(self) -> None:
        payload = self._libri()
        payload["splits"]["test"][0]["speaker"] = "0"
        unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
        payload["manifest_sha256"] = stable_json_sha256(unsigned)
        with self.assertRaisesRegex(ExperimentValidationError, "speaker leakage"):
            manifests.validate_librispeech_manifest(payload)

    def test_fleurs_reused_english_take_is_rejected(self) -> None:
        take = {"path": "x/en/dev/a.wav", "bytes": 1, "sha256": "0" * 64}
        payload = {
            "schema_version": manifests.MANIFEST_SCHEMA_VERSION,
            "dataset": "FLEURS",
            "languages": ["fr_fr"],
            "pairs": {"fr_fr": [{
                "sentence_id": "s1",
                "english_reference": take,
                "english_query": take,
                "foreign_query": {"path": "x/fr/dev/b.wav", "bytes": 1, "sha256": "1" * 64},
            }]},
        }
        payload["manifest_sha256"] = stable_json_sha256(payload)
        with self.assertRaisesRegex(ExperimentValidationError, "reuses"):
            manifests.validate_fleurs_manifest(payload)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.eye(6, dtype=np.float64)
        self.probe = self.reference + 0.01
        self.baseline = np.roll(self.reference, 1, axis=0) + 0.01

    def test_retrieval_schema_has_all_metrics_and_paired_intervals(self) -> None:
        result = evaluation.retrieval_metrics(
            self.probe,
            self.reference,
            pt_ml_probe=self.baseline,
            bootstrap_samples=50,
        )
        self.assertEqual(result["hit_count"], 6)
        self.assertEqual(result["n"], 6)
        for name in evaluation.RETRIEVAL_METRICS:
            self.assertIn(name, result)
            self.assertIn(name, result["confidence_intervals"]["difference_vs_pt_ml"])

    def test_complete_candidate_result_uses_one_contract(self) -> None:
        retrieval = {
            task: {"overall": (self.probe, self.reference, self.baseline)}
            for task in evaluation.RETRIEVAL_TASKS
        }
        result = evaluation.evaluate_candidate(
            comparison=2,
            candidate_id="direct-lambda-1",
            weight=1.0,
            precision="pre_quantization",
            english_prediction=self.probe,
            english_target=self.reference,
            pt_ml_english_prediction=self.baseline,
            retrieval_inputs=retrieval,
            diagnostic_embeddings={"candidate": self.probe},
            manifest_hashes={"librispeech": "a", "fleurs": "b"},
        )
        evaluation.validate_result(result)
        self.assertEqual(
            set(result["evaluations"]),
            {"english_voicechat_space", *evaluation.RETRIEVAL_TASKS},
        )
        self.assertIn("norm", result["embedding_diagnostics"]["candidate"])
        post = copy.deepcopy(result)
        post["precision"] = "post_quantization"
        evaluation.validate_precision_pair(result, post)


if __name__ == "__main__":
    unittest.main()
