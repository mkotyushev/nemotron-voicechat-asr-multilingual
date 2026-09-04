from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from asr_align import baseline, evaluation, export, manifests
from asr_align.experiments import EXPECTED_REPO_IDS, ExperimentValidationError
from asr_align import weights as weights_module
from asr_align.weights import EncoderWeights
import pt_ml_baseline as baseline_runner


def _weights(name: str, offset: float = 0.0) -> EncoderWeights:
    tensors = {
        "encoder.a.weight": torch.tensor([[1.0 + offset, 2.0], [3.0, 4.0]]),
        "encoder.a.bias": torch.tensor([offset, 1.0 + offset]),
    }
    return EncoderWeights(
        tensors,
        {
            "num_hidden_layers": 1,
            "hidden_size": 2,
            "num_attention_heads": 1,
            "intermediate_size": 2,
            "num_mel_bins": 2,
            "conv_kernel_size": 3,
            "sliding_window": 57,
        },
        name,
    )


def _ft_weights() -> EncoderWeights:
    value = _weights("ft", 2.0)
    value.update(
        {
            "proj.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "proj.bias": torch.tensor([0.5, -0.5]),
            "featurizer.fb": torch.ones(2, 3),
            "featurizer.window": torch.ones(4),
        }
    )
    return value


class BaselineInvariantTests(unittest.TestCase):
    def test_attach_uses_only_pt_ml_encoder_and_exact_ft_interface(self) -> None:
        pt_ml, ft_en = _weights("ml"), _ft_weights()
        attached = baseline.attach_voicechat_interface(pt_ml, ft_en)
        self.assertEqual(
            set(attached), {"encoder.a.weight", "encoder.a.bias", *baseline.ATTACHED_KEYS}
        )
        self.assertEqual(attached.config, pt_ml.config)
        self.assertIsNot(attached.config, pt_ml.config)
        baseline.assert_exact_tensors(pt_ml, attached, keys=list(pt_ml))
        baseline.assert_exact_tensors(ft_en, attached, keys=baseline.ATTACHED_KEYS)
        pt_ml["encoder.a.bias"].add_(10)
        ft_en["proj.bias"].add_(10)
        self.assertFalse(torch.equal(pt_ml["encoder.a.bias"], attached["encoder.a.bias"]))
        self.assertFalse(torch.equal(ft_en["proj.bias"], attached["proj.bias"]))

    def test_exact_tensor_check_rejects_value_shape_and_key_changes(self) -> None:
        expected = {"encoder.x": torch.ones(2)}
        baseline.assert_exact_tensors(expected, {"encoder.x": torch.ones(2)})
        with self.assertRaisesRegex(ExperimentValidationError, "changed"):
            baseline.assert_exact_tensors(expected, {"encoder.x": torch.tensor([1.0, 2.0])})
        with self.assertRaisesRegex(ExperimentValidationError, "shape"):
            baseline.assert_exact_tensors(expected, {"encoder.x": torch.ones(1, 2)})
        with self.assertRaisesRegex(ExperimentValidationError, "keys"):
            baseline.assert_exact_tensors(expected, {})

    def test_numeric_change_forbids_broadcasting(self) -> None:
        report = baseline.numeric_change(torch.ones(2), torch.tensor([1.0, 2.0]))
        self.assertEqual(report["changed_values"], 1)
        self.assertEqual(report["max_abs"], 1.0)
        with self.assertRaisesRegex(ExperimentValidationError, "shapes differ"):
            baseline.numeric_change(torch.ones(2), torch.ones(1, 2))

    def test_deterministic_finite_forward_check(self) -> None:
        class Tiny(torch.nn.Module):
            def forward(self, value):
                return value * 2

            def project(self, value):
                return value + 1

        report, hidden, projected = baseline.deterministic_forward_check(
            Tiny(), torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
        )
        self.assertTrue(report["deterministic"])
        self.assertTrue(torch.equal(hidden + 1, projected))


class BaselineEvaluationTests(unittest.TestCase):
    def _result(self, precision: str, probe: np.ndarray) -> dict:
        reference = np.eye(6, dtype=np.float64)
        retrieval = {
            task: {"overall": (probe, reference, probe)} for task in evaluation.RETRIEVAL_TASKS
        }
        return evaluation.evaluate_candidate(
            comparison=1,
            candidate_id=baseline.BASELINE_CANDIDATE_ID,
            weight=0.0,
            precision=precision,
            english_prediction=probe,
            english_target=reference,
            pt_ml_english_prediction=probe,
            retrieval_inputs=retrieval,
            diagnostic_embeddings={"candidate": probe},
            manifest_hashes={"librispeech": "a", "fleurs": "b"},
        )

    def test_baseline_is_its_own_exact_paired_reference(self) -> None:
        probe = np.eye(6, dtype=np.float64) + 0.01
        result = self._result("pre_quantization", probe)
        baseline.assert_zero_reference_deltas(result)

    def test_precision_delta_pairs_the_same_candidate(self) -> None:
        pre = self._result("pre_quantization", np.eye(6) + 0.01)
        post = self._result("post_quantization", np.eye(6) + 0.02)
        delta = baseline.precision_metric_delta(pre, post)
        self.assertEqual(delta["direction"], "post_quantization - pre_quantization")
        self.assertIn("english_voicechat_space", delta)

    def test_runner_uses_distinct_english_takes_and_intrinsic_hidden_space(self) -> None:
        matrix = np.eye(6, dtype=np.float32) + 0.01
        libri = {
            "pre_hidden_pooled": matrix,
            "pre_english_prediction": matrix,
            "english_target": matrix,
        }
        fleurs = {
            "fr_fr": {
                "pre_english_query_voicechat": matrix,
                "pre_foreign_voicechat": matrix,
                "pre_foreign_hidden": matrix,
                "pre_english_reference_hidden": matrix,
                "target_english_reference_voicechat": matrix,
            }
        }
        result = baseline_runner._evaluate_stage(
            "pre_quantization",
            libri,
            fleurs,
            manifest_hashes={"librispeech": "a", "fleurs": "b"},
            seed=0,
        )
        self.assertEqual(
            set(result["evaluations"]["candidate_on_english_retrieval"]["groups"]),
            {"fr_fr"},
        )
        baseline.assert_zero_reference_deltas(result)


class BaselineExportTests(unittest.TestCase):
    def test_baseline_export_does_not_claim_an_alignment_map(self) -> None:
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
            output = root / "baseline"
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
                    "artifact_kind": "pt_ml_baseline",
                    "projection_dim": 2,
                    "shared_setup_sha256": "abc",
                },
            )
            written_config = json.loads((output / "config.json").read_text())
            self.assertIn("voicechat_baseline", written_config)
            self.assertNotIn("voicechat_alignment", written_config)
            stripped = dict(written_config)
            stripped.pop("derived_from")
            stripped.pop("voicechat_baseline")
            self.assertEqual(stripped, config)
            self.assertTrue((output / "baseline.json").is_file())
            self.assertFalse((output / "alignment.json").exists())
            card = (output / "README.md").read_text()
            self.assertIn("No alignment map was fitted or applied", card)


class MmprojReloadTests(unittest.TestCase):
    def test_q8_0_simulator_uses_float32_and_roundf_semantics(self) -> None:
        # With max=127 the scale is exactly one, so +/-0.5 isolates C roundf's
        # half-away-from-zero behavior from scale rounding.
        value = torch.zeros(32, dtype=torch.float32)
        value[0], value[1], value[2] = 127.0, 0.5, -0.5
        value[3] = torch.nextafter(torch.tensor(0.5), torch.tensor(0.0))
        observed = weights_module.q8_0(value)
        self.assertEqual(observed[0].item(), 127.0)
        self.assertEqual(observed[1].item(), 1.0)
        self.assertEqual(observed[2].item(), -1.0)
        self.assertEqual(observed[3].item(), 0.0)

    def test_actual_mmproj_names_are_restored_to_canonical_shapes(self) -> None:
        class FakeSource:
            def __init__(self):
                self.names = []

            def f32(self, name):
                self.names.append(name)
                if name.startswith("a.conv1d.") and name.endswith(".bias"):
                    return np.arange(2, dtype=np.float32).reshape(2, 1, 1)
                if name.endswith("conv_pw1.weight") or name.endswith("conv_pw2.weight"):
                    return np.ones((2, 2), dtype=np.float32)
                if name.endswith("conv_dw.weight"):
                    return np.ones((2, 3), dtype=np.float32)
                if name == "a.mel_filters":
                    return np.ones((weights_module.N_MEL, 3), dtype=np.float32)
                if name == "a.window":
                    return np.ones(4, dtype=np.float32)
                if name.endswith(".bias") or name.endswith("pos_bias_u") or name.endswith("pos_bias_v"):
                    return np.ones(2, dtype=np.float32)
                return np.ones((2, 2), dtype=np.float32)

        source = FakeSource()
        config = {"num_hidden_layers": 1}
        with mock.patch.object(weights_module, "_gguf_source", return_value=source):
            loaded = weights_module.load_mmproj(Path("baseline-Q8_0.gguf"), Path("work"), config=config)
        self.assertEqual(loaded["encoder.subsampling.conv_in.bias"].shape, (2,))
        self.assertEqual(loaded["encoder.layers.0.conv.pointwise_conv1.weight"].shape, (2, 2, 1))
        self.assertEqual(loaded["encoder.layers.0.conv.depthwise_conv.weight"].shape, (2, 1, 3))
        self.assertIn("mm.a.proj.weight", source.names)
        self.assertIn("a.blk.0.attn_q.weight", source.names)
        self.assertEqual(loaded.config, config)
        self.assertIsNot(loaded.config, config)


class VoicechatSafetensorsTests(unittest.TestCase):
    def test_full_precision_voicechat_source_maps_ne_mo_names(self) -> None:
        class FakeSafeTensors:
            def __init__(self):
                self.names = []

            def __contains__(self, name):
                return name.startswith(weights_module.CONTAINER_PREFIX) and ".layers.1." not in name

            def f32(self, name):
                self.names.append(name)
                if name.endswith("preprocessor.featurizer.fb"):
                    return np.ones((1, weights_module.N_MEL, 257), dtype=np.float32)
                if name.endswith("preprocessor.featurizer.window"):
                    return np.ones(512, dtype=np.float32)
                if name.endswith("proj.weight"):
                    return np.full((4, 2), 0.12345, dtype=np.float32)
                if name.endswith("proj.bias"):
                    return np.ones(4, dtype=np.float32)
                if name.endswith("conv.depthwise_conv.weight"):
                    return np.ones((2, 1, 3), dtype=np.float32)
                if name.endswith(".bias") or name.endswith("pos_bias_u") or name.endswith("pos_bias_v"):
                    return np.ones(2, dtype=np.float32)
                return np.ones((2, 2), dtype=np.float32)

        source = FakeSafeTensors()
        with mock.patch.object(weights_module, "_safetensors", return_value=source):
            loaded = weights_module.load_voicechat_safetensors(Path("voicechat/model.safetensors"))
        self.assertEqual(loaded.n_layer, 1)
        self.assertEqual(loaded["proj.weight"].shape, (4, 2))
        self.assertEqual(loaded["featurizer.fb"].shape, (weights_module.N_MEL, 257))
        self.assertAlmostEqual(float(loaded["proj.weight"][0, 0]), 0.12345, places=6)
        self.assertIn(
            "stt_model.perception.encoder.layers.0.self_attn.linear_q.weight", source.names
        )


class SharedSetupConsumerTests(unittest.TestCase):
    def test_load_shared_setup_rechecks_manifest_hashes_and_languages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            libri_payload = {
                "schema_version": manifests.MANIFEST_SCHEMA_VERSION,
                "dataset": "LibriSpeech",
                "root": str(root / "libri"),
                "seed": 7,
                "splits": {
                    split: [{
                        "path": f"{index}/1/{index}.flac",
                        "speaker": str(index),
                        "chapter": "1",
                        "offset": 0,
                        "n_samples": 16000,
                        "bytes": 1,
                        "sha256": "0" * 64,
                    }]
                    for index, split in enumerate(manifests.LIBRISPEECH_SPLITS)
                },
            }
            libri_path = root / "librispeech.json"
            libri = manifests.write_frozen(libri_path, libri_payload)
            take = {"path": "x/en/dev/a.wav", "bytes": 1, "sha256": "0" * 64}
            fleurs_payload = {
                "schema_version": manifests.MANIFEST_SCHEMA_VERSION,
                "dataset": "FLEURS",
                "root": str(root / "fleurs"),
                "seed": 7,
                "languages": sorted(baseline.REQUIRED_FLEURS_LANGUAGES),
                "pairs": {
                    language: [{
                        "sentence_id": "s1",
                        "english_reference": take,
                        "english_query": {
                            "path": "x/en/dev/b.wav", "bytes": 1, "sha256": "1" * 64
                        },
                        "foreign_query": {
                            "path": f"x/{language}/dev/c.wav",
                            "bytes": 1,
                            "sha256": "2" * 64,
                        },
                    }]
                    for language in baseline.REQUIRED_FLEURS_LANGUAGES
                },
            }
            fleurs_path = root / "fleurs.json"
            fleurs = manifests.write_frozen(fleurs_path, fleurs_payload)
            setup = {
                "schema_version": "1.1",
                "roles": {"E": "PT_EN", "M": "PT_ML", "F": "FT_EN"},
                "checkpoints": {
                    role: {
                        "name": {"E": "PT_EN", "M": "PT_ML", "F": "FT_EN"}[role],
                        "repo_id": EXPECTED_REPO_IDS[role],
                        "revision": role.lower() * 8,
                        "path": str(root / role),
                        "kind": "voicechat_safetensors" if role == "F" else "asr",
                        "files": [{
                            "path": str(root / f"{role}.bin"),
                            "bytes": 1,
                            "sha256": "0" * 64,
                        }],
                        "configuration_sha256": "config-m" if role == "M" else f"config-{role}",
                    }
                    for role in ("E", "M", "F")
                },
                "candidate_runtime_configuration": {
                    "source": "M/PT_ML",
                    "source_configuration_sha256": "config-m",
                    "exact_match": True,
                },
                "precision": {
                    "quantization_stage": "final_artifact_only",
                    "deployment_quantization": "Q8_0",
                    "ft_en_source_note": (
                        "original VoiceChat safetensors; no deployment quantization in the source"
                    ),
                },
                "manifests": {
                    "librispeech": {"path": str(libri_path), "sha256": libri["manifest_sha256"]},
                    "fleurs": {"path": str(fleurs_path), "sha256": fleurs["manifest_sha256"]},
                },
            }
            setup_path = root / "shared_setup.json"
            setup_path.write_text(json.dumps(setup), encoding="utf-8")
            loaded = baseline.load_shared_setup(setup_path, verify_checkpoint_hashes=False)
            self.assertEqual(loaded.seed, 7)
            self.assertEqual(loaded.deployment_quantization, "Q8_0")


if __name__ == "__main__":
    unittest.main()
