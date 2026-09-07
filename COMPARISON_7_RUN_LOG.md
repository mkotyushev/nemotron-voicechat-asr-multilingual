# Comparison 7 run log: end-to-end interface fitting, and the merge ablation

**Status: in progress. No arm has been fitted and no result is claimed.** This
file records what is frozen, what has been verified, and what the next stage
is, so the run can be resumed without re-deriving any of it. Measurements
belong in `COMPARISON_7_RESULTS.md` when there are any; the §11 gating checks
that unblocked this comparison are already in `COMPARISON_7_GATE_RESULTS.md`.

Experiment output directory: `.cache/experiments/comparison-7-interface-v1`.

## Frozen inputs

| Record | SHA-256 |
|---|---|
| `shared-v1/shared_setup.json` | `37a872099a0b208a…` |
| `dataset-b-v1/dataset_b.json` | `f8595541d97b4200…` |
| `comparison-7-interface-v1/experiment.json` | `6432d20a11a4953b…` |

`experiment.json` pins, per arm, the encoder source and the initialization the
projection starts from, which is what invariant 6 requires a gradient-fitted
projection to record:

| Arm | Encoder source | Projection initialization |
|---|---|---|
| E1 | `FT_EN` VoiceChat fp32 `d553750c…` | the original VoiceChat projection |
| E2 | `PT_ML` `9eebdd65…` | comparison 3's folded map `b5dc62ac…` |
| E3 | comparison 6 `regmean-plus-plus` `0c2ce97e…` | the original projection |
| E4 | comparison 6 `simple-average` `413f93b1…` | the original projection |

It reproduces: recomputing `prepare` from the same inputs into a scratch
directory produced a byte-identical `experiment.json`, and `write_frozen`
accepts the identical rerun in place.

## Dataset B, verified independently of the code that built it

7,043 recordings / 7.2 hours, whole commands, no truncation:

| split | B1 en | B2 fr | B2 de | B2 ru |
|---|---:|---:|---:|---:|
| train | 1,830 | 1,503 | 1,503 | 1,503 |
| validation | 203 | 167 | 167 | 167 |

- **Disjoint from Dataset A**: B1 ∩ SLURP-A = ∅ (2,033 against 559),
  B2 ∩ Speech-MASSIVE-A = ∅ (1,670 against 263).
- **Disjoint from the §11 gating sample**: B2 ∩ 100 gated utterances = ∅.
- **No split leakage**: zero utterances appear in both train and validation, so
  no locale of a translated utterance crosses the boundary.
- **Reserved partitions untouched**: only SLURP `devel` and Speech-MASSIVE
  `dev` appear.
- **Budgets nested** 25% ⊂ 50% ⊂ 100% (1,582 ⊂ 3,168 ⊂ 6,339), every draw a
  training clip.
- **Slot replacement methods recorded**: 3,393 of 5,010 B2 rows carry at least
  one MASSIVE per-slot annotation (`translation`, `localization`, `unchanged`,
  `unchanged_translation`), which is what identifies where the two output
  conditions are most likely to diverge.

Both output-language prompts are frozen and pinned identically in the dataset
manifest and in `experiment.json`, as two conditions rather than a column.

## Stages, and where the run is

| Stage | State |
|---|---|
| `prepare` | frozen, reproducible |
| `cache` (frozen encoder outputs) | E1 frozen: 7,043 clips, 1.4 GB, encoder digest `5c2e4dd7…`. E2/E3/E4 running |
| `targets-text` (B2) | running, ~5.7 clips/min over 5,010 clips under both prompts |
| `targets-audio` (B1) | not started — needs the runtime container on the **original** perception mmproj |
| `freeze-targets` | blocked on both target pools |
| `fit` | blocked on targets; E1 must run first |
| `gate` | implemented this session; blocked on E1's fit |
| `export` | implemented this session; blocked on a fit |
| shared evaluation, MASSIVE `test` scoring, quantization, speech-to-action | not started |

The B2 teacher decodes greedily (`temperature: 0`) against a stock llama.cpp
server on the Q8_0 STT LM, one request at a time, resuming per clip and prompt
from content-addressed files. A spot check of the first clips shows prompt A
answering in English and prompt B in German on the same German command, both
passing the frozen output-language gate.

## Fixed this session

- **The encoder cache stage never ran.** `features.log_mel` computes its STFT
  in float64 for runtime parity and every other caller lands it back on F32
  with `.float()`; the Comparison 7 cache stage did not, so the first clip died
  in `conv2d` with `Input type (double) and bias type (float)`. Fixed, and the
  stage now also rejects a sample rate that is not the frozen featurizer's.
- **Nothing wrote `E1_english_gate.json`**, which `fit` requires before any arm
  other than E1 may run. Added the `gate` stage: it scores the held-out English
  B1 clips under the fitted projection and under E1's own initialization —
  which is the untouched `FT_EN` projection — and requires that fitting has not
  cost more than `GATE_TOLERANCE_NATS` (0.05) of token cross-entropy in *any*
  prompt cell. Per cell, not on the average: a small cell can regress badly
  while a clip-weighted mean still improves, and the unit test asserts exactly
  that case fails. This is the design record's §9 pipeline check; it certifies
  the training loop, not the deployment endpoint, and the `FT_EN` control row
  of the speech-to-action table remains a separate later check.
- **Nothing exported a fitted arm.** Added the `export` stage, writing the
  directory `convert_asr_to_mmproj.py --asr-dir` can consume alone, and giving
  `asr_align/export.py` a `gradient_fitted_interface` branch and model card so
  a Comparison 7 artifact describes itself. It reloads what it wrote and
  asserts the encoder is byte-identical to the arm's source and the exported
  interface equals the fit, which is invariant 6's "nothing else was trained"
  as a check rather than a claim.

## Resume commands

```bash
# encoder caches (one arm at a time; each holds ~3 GB next to the teachers)
.venv-align/bin/python interface_fitting.py cache --arm E2 \
  --output .cache/experiments/comparison-7-interface-v1 --device cuda

# B2 targets, resumable per clip and prompt
.venv-align/bin/python interface_fitting.py targets-text \
  --endpoint http://127.0.0.1:9099 \
  --teacher-model /srv/bulk/ai/models/NemotronLabs-VoiceChat-11B-gguf/nemotron_voicechat_11b-stt-llm-Q8_0.gguf \
  --teacher-binary .cache/tools/llama.cpp/llama-b10819/llama-server \
  --massive .cache/datasets/MASSIVE/1.1/data \
  --output .cache/experiments/comparison-7-interface-v1

# B1 targets: the container must serve the ORIGINAL perception encoder, not a
# candidate, and VC_DUMP=1 must reach it (asr_align/gating.py injects it).
.venv-align/bin/python interface_fitting.py targets-audio \
  --massive .cache/datasets/MASSIVE/1.1/data \
  --output .cache/experiments/comparison-7-interface-v1

.venv-align/bin/python interface_fitting.py freeze-targets \
  --output .cache/experiments/comparison-7-interface-v1
.venv-align/bin/python interface_fitting.py fit --arm E1 --budget 100 \
  --output .cache/experiments/comparison-7-interface-v1
.venv-align/bin/python interface_fitting.py gate \
  --output .cache/experiments/comparison-7-interface-v1
```

## Open items the next session should not rediscover

- **The GPU is oversubscribed.** A Comparison 6 deployment container
  (`mmproj-asr-regmean-plus-plus-Q8_0.gguf`) has held 14.6 GB since that
  comparison finished, and the B2 text teacher holds 6 GB. The encoder caches
  fit in what is left, but §10 budgets ~8 GB for NF4 fitting, so **both must be
  stopped before the first `fit`**, and B1 target generation needs the
  container restarted on `mmproj-voicechat-perception-Q8_0.gguf` anyway.
- **B1 is the long pole, not B2.** 2,033 clips under two prompts is 4,066
  runtime turns, each streaming its audio and then decoding, against B2's
  ~14 hours. Budget for it before starting.
- **The precision variable is unmeasured.** Every fit records its
  language-model precision, but the NF4-versus-bf16 gap invariant 3 asks for
  has not been run.
- `asr_align/regmean.ARTIFACT_KIND` is `"regmean-merge"` while
  `asr_align/export.py` branches on `"regmean_merge"`, so comparison 6's
  artifacts fell through to the generic `voicechat_alignment` branch and carry
  an `alignment.json` rather than the intended self-description. Their
  `model.safetensors` and `regmean_merge.json` are unaffected, and those are
  what comparison 7 and the deployment converter consume, so this is cosmetic
  and was left alone rather than rewriting a frozen artifact. Comparison 7's
  kind string matches its branch.
