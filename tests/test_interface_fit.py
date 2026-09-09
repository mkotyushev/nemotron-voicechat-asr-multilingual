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
from asr_align.frozen_lm import FITTING_GRAPH_VERSION, FrozenBlock, FrozenLinear, FrozenVoiceChatLM, fusion_weights_from_config, ssd_scan
from asr_align.interface_fit import (
    calibrate_pool_weights,
    duplex_inputs,
    duplex_turn_rejections,
    english_gate_verdict,
    parse_audio_trace,
    parse_duplex_trace,
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
    def test_fusion_comes_from_original_voicechat_config_and_rejects_missing_weights(self):
        stt = {"duplex_text_channel_weight": 1.0, "duplex_user_channel_weight": 1.0,
               "duplex_function_channel_weight": 2.0}
        self.assertEqual(fusion_weights_from_config({"model": {"stt": {"model": stt}}}),
                         {"text": 1.0, "audio": 1.0, "function": 2.0})
        for config in ({}, {"model": {"stt": {"model": {**stt, "duplex_function_channel_weight": float("nan")}}}}):
            with self.assertRaises(ExperimentValidationError):
                fusion_weights_from_config(config)

    def test_system_prefix_uses_the_runtime_weighted_duplex_equation(self):
        class TinyLM(FrozenVoiceChatLM):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.fusion_weights = {"text": 1.0, "audio": 1.0, "function": 2.0}

            def embed(self, ids):
                return ids[..., None].to(torch.bfloat16).expand(*ids.shape, 2)

            def forward(self, inputs, **kwargs):
                self.observed = inputs
                return inputs, ["prefix"]

        lm = TinyLM()
        self.assertEqual(lm.cache_prompt([1, 42, 2]), ["prefix"])
        # Runtime starts text at BOS, then holds both output channels at PAD.
        # Each frame is 1*text + 2*function + 1*conditioning audio/text.
        expected = torch.tensor([[[26., 26.], [78., 78.], [38., 38.]]])
        torch.testing.assert_close(lm.observed, expected, rtol=0, atol=0)
        self.assertEqual(lm.observed.dtype, torch.float32)

    def test_legacy_experiment_cannot_resume_after_fusion_fix(self):
        from interface_fitting import validate_fitting_graph
        with self.assertRaisesRegex(ExperimentValidationError, "prepare a new experiment"):
            validate_fitting_graph({})

    def test_resume_checkpoint_is_bound_to_its_graph_and_candidate(self):
        from interface_fitting import validate_resume_checkpoint
        good = {"fitting_graph_version": FITTING_GRAPH_VERSION, "provenance_sha256": "candidate-a"}
        validate_resume_checkpoint(good, "candidate-a")
        for saved in ({}, {**good, "fitting_graph_version": "legacy"},
                      {**good, "provenance_sha256": "candidate-b"}):
            with self.assertRaises(ExperimentValidationError):
                validate_resume_checkpoint(saved, "candidate-a")

    def test_student_duplex_timeline_uses_previous_tokens_and_zero_post_audio(self):
        class Embeddings:
            fusion_weights = {"text": 1.0, "audio": 1.0, "function": 2.0}
            fuse = FrozenVoiceChatLM.fuse

            def embed(self, ids):
                return ids[..., None].float().expand(*ids.shape, 2)
        proj = torch.nn.Linear(2, 2)
        with torch.no_grad():
            proj.weight.copy_(torch.eye(2))
            proj.bias.fill_(7)
        # Row j of the cache projects to j + 7, so the assertions below say
        # which cached frame each timeline position actually read.
        cached = torch.arange(20).float()[:, None].expand(20, 2)
        timeline = text_target_timeline([42, 43], 2)
        self.assertEqual(timeline["audio_indices"][:3], [1, 2, 3])
        inputs, labels = duplex_inputs(proj, Embeddings(), cached, timeline)
        # Two PAD channels weigh 12 + 2 * 12; the audio is cache row 1.
        torch.testing.assert_close(inputs[0, 0], torch.tensor([44., 44.]))
        # The first post-command frame is not blank any more: it reads the
        # cache's encoded silence, bias and all, which is what the Realtime
        # bridge feeds.  Under the v1 zero tail this frame was 36.
        torch.testing.assert_close(inputs[0, 2], torch.tensor([46., 46.]))
        self.assertEqual(labels[2].item(), 1)
        self.assertEqual(labels[3].item(), 42)

    def test_duplex_inputs_refuse_a_timeline_with_frames_that_hear_nothing(self):
        proj = torch.nn.Linear(2, 2)
        timeline = {"text_tokens": [12, 1], "function_tokens": [12, 12], "audio_indices": [0, -1]}
        with self.assertRaises(ExperimentValidationError) as caught:
            duplex_inputs(proj, None, torch.ones(4, 2), timeline)
        self.assertIn("encoded silence", str(caught.exception))

    def test_duplex_audio_indices_are_contiguous_and_skip_the_lead(self):
        self.assertEqual(interface_fit.duplex_audio_indices(3, offset=1), [1, 2, 3])
        # B2's timeline starts at the command, so it steps over the cache's
        # lead silence; B1's trace covers the lead and does not.
        self.assertEqual(interface_fit.duplex_audio_indices(3, start=8, offset=1), [9, 10, 11])
        with self.assertRaises(ExperimentValidationError):
            interface_fit.duplex_audio_indices(0)

    def test_audio_trace_requires_complete_frames_and_rejects_untraced_splices(self):
        trace = "\n".join(f"DUMP t={i} txt={token} 'x' top=0 fn=12" for i, token in enumerate([12, 12, 1, 42, 2]))
        timeline = parse_audio_trace(trace, prefix_frames=2, audio_frames=2)
        self.assertEqual(timeline["text_tokens"], [1, 42, 2])
        self.assertEqual(timeline["audio_indices"], [0, 1, -1])
        with self.assertRaises(ExperimentValidationError):
            parse_audio_trace(trace.replace("t=3", "t=8"), prefix_frames=2, audio_frames=2)
        with self.assertRaises(ExperimentValidationError):
            parse_audio_trace(trace.replace("fn=12", "fn=20"), prefix_frames=2, audio_frames=2)

    def test_duplex_trace_keeps_function_activity_and_records_no_audio_indices(self):
        # The duplex timeline has an encoder frame at every position -- the
        # command, then encoded silence -- so it carries no -1 audio index, and
        # unlike run_turn it does not reject spontaneous function tokens.
        trace = "\n".join(f"DUMP t={i} txt={token} 'x' top=0 fn={fn}"
                          for i, (token, fn) in enumerate([(12, 12), (12, 12), (1, 12), (42, 20), (2, 12)]))
        timeline = parse_duplex_trace(trace, prefix_frames=2)
        self.assertEqual(timeline["text_tokens"], [1, 42, 2])
        self.assertEqual(timeline["function_tokens"], [12, 20, 12])
        self.assertNotIn("audio_indices", timeline)
        with self.assertRaises(ExperimentValidationError):
            parse_duplex_trace(trace.replace("t=3", "t=8"), prefix_frames=2)

    def test_duplex_trace_drops_the_previous_turns_unflushed_frames(self):
        # The runtime's stderr is block buffered and one file carries the whole
        # session, so a slice taken by byte offset can open with frames the
        # previous turn had not written yet: about one target in seven.
        turn = [(12, 12), (12, 12), (1, 12), (42, 12), (2, 12)]
        body = "\n".join(f"DUMP t={i} txt={token} 'x' top=0 fn={fn}"
                         for i, (token, fn) in enumerate(turn))
        stale = "DUMP t=121 txt=2 'x' top=0 fn=12\nDUMP t=122 txt=12 'x' top=0 fn=12\n"
        timeline = parse_duplex_trace(stale + body, prefix_frames=2)
        self.assertEqual(timeline["text_tokens"], [1, 42, 2])
        # A hole inside this turn is still a rejection, stale head or not.
        with self.assertRaises(ExperimentValidationError):
            parse_duplex_trace(stale + body.replace("t=3", "t=9"), prefix_frames=2)
        # And a slice that never reaches the prompt's last frame is not a turn.
        with self.assertRaises(ExperimentValidationError):
            parse_duplex_trace(stale, prefix_frames=2)

    def test_duplex_rejects_barge_in_but_tolerates_boundary_slop(self):
        tolerance = interface_fit.DUPLEX_ONSET_TOLERANCE_FRAMES
        # Opening a frame or two before the encoder formally consumed the
        # command is boundary slop; those replies are complete and on topic.
        self.assertEqual(duplex_turn_rejections(opened=True, onset=-tolerance, spoken=9), [])
        self.assertEqual(duplex_turn_rejections(opened=True, onset=4, spoken=9), [])
        # Opening well inside the command is barge-in, which stays FT_EN's.
        self.assertTrue(duplex_turn_rejections(opened=True, onset=-tolerance - 1, spoken=9))
        self.assertTrue(duplex_turn_rejections(opened=True, onset=None, spoken=9))
        self.assertTrue(duplex_turn_rejections(opened=False, onset=None, spoken=None))
        self.assertTrue(duplex_turn_rejections(
            opened=True, onset=interface_fit.DUPLEX_MAX_ONSET_FRAMES + 1, spoken=9))
        # A turn that opened and then said nothing supervises nothing.
        self.assertTrue(duplex_turn_rejections(opened=True, onset=4, spoken=0))

    def test_free_running_turn_opens_on_content_and_not_on_a_stray_eos(self):
        class ScriptedLM:
            """Decodes a fixed token sequence, so only the loop is under test."""

            fusion_weights = {"text": 1.0, "audio": 1.0, "function": 2.0}
            fuse = FrozenVoiceChatLM.fuse

            def __init__(self, script):
                self.script, self.step = list(script), 0

            def embed(self, ids):
                return ids[..., None].float().expand(*ids.shape, 2)

            def __call__(self, inputs, *, prefix=None, return_state=False):
                return (inputs, prefix) if return_state else inputs

            def head(self, hidden):
                token = self.script[min(self.step, len(self.script) - 1)]
                self.step += 1
                logits = torch.full((hidden.shape[0], 64), -10.0)
                logits[:, token] = 10.0
                return logits

        proj = torch.nn.Linear(2, 2)
        cached = torch.ones(16, 2)
        timeline = {"text_tokens": [12] * 6, "function_tokens": [12] * 6,
                    "audio_indices": interface_fit.duplex_audio_indices(6)}
        # Pad, pad, pad, then two content tokens and a stop: the command runs
        # out after frame 2, so this turn opens one frame past it.
        run = interface_fit.duplex_free_run(proj, ScriptedLM([12, 12, 12, 42, 43, 2]), cached,
                                            timeline, None, command_frames=2)
        self.assertTrue(run["opened"])
        self.assertEqual(run["onset_frames_past_command"], 1)
        self.assertEqual(run["reply_token_ids"], [42, 43])
        # It stops at the EOS that closes its own turn rather than decoding
        # the whole silence tail.
        self.assertEqual(run["frames_decoded"], 6)
        # An EOS while no turn is open closes nothing and opens nothing, which
        # is the bug that made every free-running turn end at frame 9.
        silent = interface_fit.duplex_free_run(proj, ScriptedLM([12, 2, 12, 12, 12, 12]), cached,
                                               timeline, None, command_frames=2)
        self.assertFalse(silent["opened"])
        self.assertIsNone(silent["onset_frames_past_command"])
        self.assertEqual(silent["frames_decoded"], 6)
        # A turn that has not opened within the window the teacher filter would
        # accept stops there rather than decoding the whole silence tail, which
        # is what makes the silent runs the expensive ones.
        capped = interface_fit.duplex_free_run(proj, ScriptedLM([12] * 6), cached, timeline, None,
                                               command_frames=2, max_onset_frames=1)
        self.assertFalse(capped["opened"])
        self.assertEqual(capped["frames_decoded"], 4)

    def test_duplex_gate_needs_a_control_that_speaks_and_a_fit_that_still_does(self):
        control = {"A": {"n": 24, "opened": 24, "onsets": [4] * 24},
                   "B": {"n": 24, "opened": 24, "onsets": [5] * 24}}
        # The v1 pilot: teacher-forced cross-entropy was fine, and free running
        # it answered none of the turns its own initialization answered.
        silent = {"A": {"n": 24, "opened": 0, "onsets": []},
                  "B": {"n": 24, "opened": 0, "onsets": []}}
        self.assertFalse(interface_fit.duplex_gate_verdict(silent, control)["passed"])
        matching = {"A": {"n": 24, "opened": 24, "onsets": [6] * 24},
                    "B": {"n": 24, "opened": 23, "onsets": [6] * 23}}
        verdict = interface_fit.duplex_gate_verdict(matching, control)
        self.assertTrue(verdict["passed"])
        self.assertTrue(verdict["control_calibrated"])
        self.assertEqual(verdict["cells"]["B"]["fitted_onsets_in_band"], 23)
        # A control that cannot open a turn either says the harness is broken,
        # so a matching fit is not evidence of anything.
        mute_control = {cell: {**value, "opened": 0, "onsets": []} for cell, value in control.items()}
        useless = interface_fit.duplex_gate_verdict(silent, mute_control)
        self.assertFalse(useless["control_calibrated"])
        self.assertFalse(useless["passed"])

    def test_candidate_provenance_hashes_the_frame_alignment_it_trained_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = manifests.write_frozen(Path(tmp) / "b.json", fixture(Path(tmp)))
            record = {"path": "x", "bytes": 1, "sha256": "a" * 64}
            common = dict(arm="E1", budget=100, manifest=manifest, source=record,
                          initialization=record,
                          language_model={**record, "fitting_precision": "nf4"},
                          targets={"teacher_paths": manifest["teacher_paths"],
                                   "manifest_sha256": "b" * 64},
                          calibration={"weights": {}}, fitting_graph={"version": 1})
            cache = {"lead_frames": 8, "tail_frames": 213, "tensor_root": "/srv/bulk",
                     "b1_duplex_provenance_sha256": "c" * 64, "padding": "lead, command, tail"}
            at_one = interface_fit.candidate_provenance(
                **common, supervision={"frame_offset": 1, "activation_cache": cache})
            at_zero = interface_fit.candidate_provenance(
                **common, supervision={"frame_offset": 0, "activation_cache": cache})
            # 80 ms of alignment is the difference between two candidates, not
            # a detail: the same targets under either offset must not collide.
            self.assertNotEqual(at_one["provenance_sha256"], at_zero["provenance_sha256"])
            with self.assertRaises(ExperimentValidationError):
                interface_fit.candidate_provenance(**common, supervision={"frame_offset": 1})

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
