from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile
import torch

from asr_align import dataset_b, export, interface_fit, manifests
from asr_align.experiments import ExperimentValidationError, sha256_file
from asr_align.frozen_lm import FrozenBlock, FrozenLinear, ssd_scan
from asr_align.interface_fit import (
    calibrate_pool_weights,
    duplex_inputs,
    english_gate_verdict,
    parse_audio_trace,
    text_target_timeline,
)


def fixture(root):
    rows = []
    for pool, langs in [("B1", ["en"]), ("B2", ["fr", "de", "ru"])]:
        for lang in langs:
            for uid in range(20):
                path = root / f"{pool}-{lang}-{uid}.flac"
                soundfile.write(path, np.zeros(160), 16000, subtype="PCM_16")
                rows.append({"pool": pool, "language": lang, "locale": {"fr": "fr-FR", "de": "de-DE", "ru": "ru-RU", "en": "en-US"}[lang],
                             "utterance_id": str(uid), "clip_id": f"{pool}/{lang}/{uid}",
                             "partition": "devel" if pool == "B1" else "dev", "slot_method": [],
                             "transcript": "native command", "intent": "alarm_set", "path": path.name,
                             "sha256": sha256_file(path), "bytes": path.stat().st_size,
                             "offset": 0, "n_samples": 160, "source_frames": 160})
    return dataset_b.build_manifest(root, records=rows, sources={}, excluded={"B1": [], "B2": []})


class DatasetBTests(unittest.TestCase):
    def test_frozen_audio_idempotence_and_changed_input_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = fixture(root)
            first = manifests.write_frozen(root / "b.json", payload)
            self.assertEqual(first, manifests.write_frozen(root / "b.json", payload))
            dataset_b.verify_audio(first)
            altered = copy.deepcopy(payload)
            altered["seed"] += 1
            with self.assertRaises(ExperimentValidationError):
                manifests.write_frozen(root / "b.json", altered)
            row = payload["splits"]["train"][0]
            (root / row["path"]).write_bytes(b"changed")
            with self.assertRaises(ExperimentValidationError):
                dataset_b.verify_audio(payload)

    def test_nested_budgets_and_no_locale_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = fixture(Path(tmp))
            budgets = [set(payload["budgets"][str(p)]) for p in dataset_b.BUDGETS]
            self.assertTrue(budgets[0] < budgets[1] < budgets[2])
            train = {(r["pool"], r["utterance_id"]) for r in payload["splits"]["train"]}
            val = {(r["pool"], r["utterance_id"]) for r in payload["splits"]["validation"]}
            self.assertFalse(train & val)
            for mutate in (lambda p: p["excluded_ids"]["B2"].extend(str(i) for i in range(20)),
                           lambda p: p["splits"]["train"][0].update(partition="test"),
                           lambda p: p["splits"]["train"][0].update(path="../escape.flac"),
                           lambda p: p["splits"]["train"][0].update(n_samples=80)):
                changed = copy.deepcopy(payload)
                changed.pop("manifest_sha256")
                mutate(changed)
                with self.assertRaises(ExperimentValidationError):
                    dataset_b.validate_manifest(changed)


class ScanTests(unittest.TestCase):
    def test_cached_blocks_match_full_sequence_and_preserve_suffix_gradients(self):
        cfg = {"hidden_size": 8, "layer_norm_epsilon": 1e-5, "num_attention_heads": 4,
               "num_key_value_heads": 2, "head_dim": 2, "mamba_num_heads": 4,
               "mamba_head_dim": 2, "n_groups": 2, "ssm_state_size": 3, "conv_kernel": 4}
        shapes = {"norm.weight": (8,), "mixer.in_proj.weight": (32, 8),
                  "mixer.out_proj.weight": (8, 8), "mixer.q_proj.weight": (8, 8),
                  "mixer.k_proj.weight": (4, 8), "mixer.v_proj.weight": (4, 8),
                  "mixer.o_proj.weight": (8, 8), "mixer.up_proj.weight": (16, 8),
                  "mixer.down_proj.weight": (8, 16), "mixer.conv1d.weight": (20, 1, 4),
                  "mixer.conv1d.bias": (20,), "mixer.dt_bias": (4,), "mixer.A_log": (4,),
                  "mixer.D": (4,), "mixer.norm.weight": (8,)}
        for kind in ("M", "*", "-"):
            torch.manual_seed(0)
            tensors = {k: torch.randn(shape) * .1 for k, shape in shapes.items()}
            block = FrozenBlock(cfg, kind, tensors.__getitem__, precision="bf16", device="cpu")
            x = torch.randn(1, 9, 8, dtype=torch.bfloat16, requires_grad=True)
            full, _ = block(x)
            with torch.no_grad():
                _, cache = block(x[:, :4])
            cached_before = None if cache is None else [s.clone() for s in cache]
            suffix, _ = block(x[:, 4:], cache)
            torch.testing.assert_close(suffix, full[:, 4:], atol=.02, rtol=.01)
            expected_grad = torch.autograd.grad(full[:, 4:].float().square().sum(), x, retain_graph=True)[0][:, 4:]
            actual_grad = torch.autograd.grad(suffix.float().square().sum(), x)[0][:, 4:]
            torch.testing.assert_close(actual_grad, expected_grad, atol=.02, rtol=.01)
            if cache is not None:
                for before, after in zip(cached_before, cache):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_scan_and_input_gradient_match_scalar_recurrence(self):
        torch.manual_seed(7)
        x = torch.randn(2, 11, 4, 3, requires_grad=True)
        dt = torch.rand(2, 11, 4) * .2
        a, d = -torch.rand(4), torch.randn(4)
        b, c = torch.randn(2, 11, 2, 5), torch.randn(2, 11, 2, 5)
        initial = torch.randn(2, 4, 3, 5)
        y, state = ssd_scan(x, dt, a, b, c, d, initial, chunk_size=4)
        reference_state, outputs = initial, []
        for index in range(11):
            bi = b[:, index].repeat_interleave(2, 1)
            ci = c[:, index].repeat_interleave(2, 1)
            reference_state = reference_state * (dt[:, index] * a).exp()[:, :, None, None]
            reference_state = reference_state + x[:, index, :, :, None] * dt[:, index, :, None, None] * bi[:, :, None, :]
            outputs.append((reference_state * ci[:, :, None, :]).sum(-1) + x[:, index] * d[None, :, None])
        reference = torch.stack(outputs, dim=1)
        torch.testing.assert_close(y, reference, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(state, reference_state, atol=3e-6, rtol=3e-6)
        expected_grad = torch.autograd.grad(reference.square().sum(), x, retain_graph=True)[0]
        grad = torch.autograd.grad(y.square().sum(), x)[0]
        torch.testing.assert_close(grad, expected_grad, atol=3e-5, rtol=3e-5)

    def test_prefix_split_matches_full_scan(self):
        torch.manual_seed(3)
        x = torch.randn(1, 15, 4, 2)
        dt, a, d = torch.rand(1, 15, 4), -torch.rand(4), torch.rand(4)
        b, c = torch.randn(1, 15, 2, 3), torch.randn(1, 15, 2, 3)
        full, _ = ssd_scan(x, dt, a, b, c, d)
        _, prefix = ssd_scan(x[:, :7], dt[:, :7], a, b[:, :7], c[:, :7], d)
        suffix, _ = ssd_scan(x[:, 7:], dt[:, 7:], a, b[:, 7:], c[:, 7:], d, prefix)
        torch.testing.assert_close(suffix, full[:, 7:], atol=2e-6, rtol=2e-6)

    def test_offloaded_linear_backpropagates_only_to_input(self):
        weight = torch.randn(5, 3)
        module = FrozenLinear(weight, precision="bf16_cpu_offload", device="cpu")
        x = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)
        module(x).float().sum().backward()
        torch.testing.assert_close(x.grad, module.weight.sum(0).expand_as(x))
        self.assertFalse(list(module.parameters()))


class SupervisionTests(unittest.TestCase):
    def test_student_duplex_timeline_uses_previous_tokens_and_zero_post_audio(self):
        class Embeddings:
            def embed(self, ids):
                return ids[..., None].float().expand(*ids.shape, 2)
        proj = torch.nn.Linear(2, 2)
        with torch.no_grad():
            proj.weight.copy_(torch.eye(2))
            proj.bias.fill_(7)
        timeline = text_target_timeline([42, 43], 2)
        inputs, labels = duplex_inputs(proj, Embeddings(), torch.ones(2, 2), timeline)
        torch.testing.assert_close(inputs[0, 0], torch.tensor([32., 32.]))
        # First post-audio frame consumes two PAD channels and NO proj.bias.
        torch.testing.assert_close(inputs[0, 2], torch.tensor([24., 24.]))
        self.assertEqual(labels[2].item(), 1)
        self.assertEqual(labels[3].item(), 42)

    def test_audio_trace_requires_complete_frames_and_rejects_untraced_splices(self):
        trace = "\n".join(f"DUMP t={i} txt={token} 'x' top=0 fn=12" for i, token in enumerate([12, 12, 1, 42, 2]))
        timeline = parse_audio_trace(trace, prefix_frames=2, audio_frames=2)
        self.assertEqual(timeline["text_tokens"], [1, 42, 2])
        self.assertEqual(timeline["audio_indices"], [0, 1, -1])
        with self.assertRaises(ExperimentValidationError):
            parse_audio_trace(trace.replace("t=3", "t=8"), prefix_frames=2, audio_frames=2)
        with self.assertRaises(ExperimentValidationError):
            parse_audio_trace(trace.replace("fn=12", "fn=20"), prefix_frames=2, audio_frames=2)

    def test_english_gate_fails_a_single_regressed_prompt_cell(self):
        initialization = {"B1/en/A_english_only": {"n": 100, "mean": .40},
                          "B1/en/B_input_language": {"n": 4, "mean": .40}}
        improved = {"B1/en/A_english_only": {"n": 100, "mean": .38},
                    "B1/en/B_input_language": {"n": 4, "mean": .42}}
        verdict = english_gate_verdict(improved, initialization)
        self.assertTrue(verdict["passed"])
        self.assertLess(verdict["delta"], 0)
        # A cell may regress well past tolerance while the clip-weighted mean
        # still improves; the gate is per cell exactly so that cannot pass.
        broken = {"B1/en/A_english_only": {"n": 100, "mean": .30},
                  "B1/en/B_input_language": {"n": 4, "mean": 2.40}}
        hidden = english_gate_verdict(broken, initialization)
        self.assertLess(hidden["delta"], 0)
        self.assertFalse(hidden["passed"])
        self.assertFalse(hidden["cells"]["B1/en/B_input_language"]["passed"])
        self.assertTrue(hidden["cells"]["B1/en/A_english_only"]["passed"])

    def test_english_gate_rejects_incomparable_measurements(self):
        base = {"B1/en/A_english_only": {"n": 2, "mean": .5}}
        for fitted, initialization in (
            ({}, {}),
            (base, {"B1/en/B_input_language": {"n": 2, "mean": .5}}),
            ({"B1/en/A_english_only": {"n": 3, "mean": .5}}, base),
            ({"B1/en/A_english_only": {"n": 2, "mean": float("nan")}}, base),
        ):
            with self.assertRaises(ExperimentValidationError):
                english_gate_verdict(fitted, initialization)

    def test_calibration_uses_measured_unequal_scales(self):
        result = calibrate_pool_weights({"B1": [2, 2, 2], "B2": [8, 8, 8]})
        self.assertAlmostEqual(result["weights"]["B1"], 1.6)
        self.assertAlmostEqual(result["weights"]["B2"], .4)
        with self.assertRaises(ExperimentValidationError):
            calibrate_pool_weights({"B1": [0], "B2": [1]})


class ExportTests(unittest.TestCase):
    def test_interface_fit_export_says_only_proj_was_trained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps(
                {"model_type": "nemotron3_5_asr", "encoder_config": {"sliding_window": 57}}))
            output = root / "E2-100-nf4"
            export.export(
                output, source=source,
                encoder={"encoder.x": np.ones((2, 2), dtype=np.float32)},
                proj_weight=np.eye(2, dtype=np.float32), proj_bias=np.zeros(2, dtype=np.float32),
                featurizer={"fb": np.ones((2, 3), dtype=np.float32), "window": np.ones(4, dtype=np.float32)},
                report={"artifact_kind": interface_fit.ARTIFACT_KIND, "arm": "E2",
                        "arm_definition": dataset_b.ARMS["E2"], "data_budget_percent": 100,
                        "fitting_precision": "nf4"})
            written = json.loads((output / "config.json").read_text())
            self.assertEqual(written["voicechat_interface_fit"]["arm"], "E2")
            self.assertEqual(written["voicechat_interface_fit"]["fitting_precision"], "nf4")
            self.assertNotIn("voicechat_alignment", written)
            self.assertTrue((output / "interface_fit.json").is_file())
            self.assertFalse((output / "alignment.json").exists())
            card = (output / "README.md").read_text()
            self.assertIn("byte-identical to its source", card)
            self.assertIn("comparison_3_folded_map", card)


if __name__ == "__main__":
    unittest.main()
