from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from asr_align import direct, evaluation, export
from asr_align.experiments import ExperimentValidationError, task_vector


class DirectArithmeticTests(unittest.TestCase):
    def test_task_vector_report_has_tensor_module_and_block_norms(self) -> None:
        e = {
            "encoder.layers.0.self_attn.q_proj.weight": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]]
            ),
            "encoder.subsampling.linear.bias": torch.tensor([1.0, 2.0]),
        }
        f = {
            "encoder.layers.0.self_attn.q_proj.weight": torch.tensor(
                [[2.0, 4.0], [6.0, 8.0]]
            ),
            "encoder.subsampling.linear.bias": torch.tensor([2.0, 4.0]),
        }
        delta = task_vector(e, f)
        report = direct.task_vector_report(e, f, delta)
        self.assertTrue(report["reconstruction"]["passed"])
        self.assertEqual(
            set(report["by_module_type"]), {"self_attention", "subsampling"}
        )
        self.assertEqual(
            set(report["by_block"]), {"encoder.layers.0", "encoder.subsampling"}
        )
        self.assertEqual(report["total"]["tensor_count"], 2)
        self.assertEqual(set(report["by_tensor"]), set(delta))

        bad = dict(delta)
        bad["encoder.subsampling.linear.bias"] = bad[
            "encoder.subsampling.linear.bias"
        ] + 1
        with self.assertRaisesRegex(ExperimentValidationError, "reconstruction"):
            direct.task_vector_report(e, f, bad)

    def test_stable_candidate_ids_and_activation_growth_tripwire(self) -> None:
        self.assertEqual(direct.candidate_id(0.25), "direct-lambda-0p25")
        self.assertEqual(direct.candidate_id(1.0), "direct-lambda-1")
        with self.assertRaises(ExperimentValidationError):
            direct.candidate_id(0.3)

        reference = direct.activation_profile({"residual": torch.ones(1, 2, 3)})
        candidate = direct.activation_profile({"residual": torch.full((1, 2, 3), 2.0)})
        report = direct.activation_growth_report(reference, candidate, maximum=3.0)
        self.assertEqual(report["maximum_observed_ratio"], 2.0)
        with self.assertRaisesRegex(ExperimentValidationError, "abnormal"):
            direct.activation_growth_report(reference, candidate, maximum=1.5)


class DirectParetoTests(unittest.TestCase):
    @staticmethod
    def _result(weight: float) -> dict:
        english = weight
        retention = 1.0 - weight
        groups = {
            "fr_fr": {"mrr": retention, "top1": retention},
            "de_de": {"mrr": retention, "top1": retention},
        }
        return {
            "lambda": weight,
            "candidate_id": direct.candidate_id(weight),
            "precision": "pre_quantization",
            "evaluations": {
                "english_voicechat_space": {"r2": english, "cosine_mean": english},
                "intrinsic_candidate_crosslingual_retrieval": {"groups": groups},
                "historical_centered_fleurs_retrieval": {"groups": groups},
            },
        }

    def test_pareto_table_preserves_full_sweep_without_selecting(self) -> None:
        report = direct.pareto_table(
            [self._result(weight) for weight in (0.0, 0.25, 0.5, 0.75, 1.0)]
        )
        self.assertFalse(report["selection_performed"])
        self.assertEqual(len(report["rows"]), 5)
        self.assertTrue(all(row["pareto_efficient"] for row in report["rows"]))


class DirectExportTests(unittest.TestCase):
    def test_direct_export_records_arithmetic_without_claiming_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            config = {
                "model_type": "nemotron3_5_asr",
                "encoder_config": {"sliding_window": 57},
            }
            (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (source / "processor_config.json").write_text("{}\n", encoding="utf-8")
            output = root / "direct"
            export.export(
                output,
                source=source,
                encoder={"encoder.x": np.ones((2, 2), dtype=np.float32)},
                proj_weight=np.eye(2, dtype=np.float32),
                proj_bias=np.zeros(2, dtype=np.float32),
                featurizer={
                    "fb": np.ones((2, 3), dtype=np.float32),
                    "window": np.ones(4, dtype=np.float32),
                },
                report={
                    "artifact_kind": direct.ARTIFACT_KIND,
                    "lambda": 0.5,
                },
            )
            written = json.loads((output / "config.json").read_text())
            self.assertEqual(written["voicechat_direct_task_arithmetic"]["lambda"], 0.5)
            self.assertNotIn("voicechat_alignment", written)
            self.assertTrue((output / "direct_task_arithmetic.json").is_file())
            self.assertFalse((output / "alignment.json").exists())
            self.assertIn("No activation or interface map was fitted", (output / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
