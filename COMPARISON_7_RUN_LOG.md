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
| `cache` (frozen encoder outputs) | **all four arms frozen**, 7,043 clips and 1.4 GB each |
| `targets-text` (B2) | running, ~5.7 clips/min over 5,010 clips under both prompts |
| `targets-audio` (B1) | path exercised on 24 clips in scratch; the real pool is not started |
| `freeze-targets` | blocked on both target pools |
| `fit` | blocked on targets; E1 must run first |
| `gate` | implemented this session; blocked on E1's fit |
| `export` | implemented **and exercised** on a scratch E1 fit; blocked on a real one |
| shared evaluation, MASSIVE `test` scoring, quantization, speech-to-action | not started |

The frozen encoder outputs, one cache per arm over the same 7,043 clips:

| Arm | encoder digest |
|---|---|
| E1 | `5c2e4dd741de22fb…` |
| E2 | `a5a4be667b089003…` |
| E3 | `7c8e188db0d0524d…` |
| E4 | `ba705d67d77d9fc3…` |

The B2 teacher decodes greedily (`temperature: 0`) against a stock llama.cpp
server on the Q8_0 STT LM, one request at a time, resuming per clip and prompt
from content-addressed files. A spot check of the first clips shows prompt A
answering in English and prompt B in German on the same German command, both
passing the frozen output-language gate.

## The fitting loop, measured

`fit` cannot run until both teacher pools are frozen, but its inner loop can be
exercised on what already exists: real E1 cached activations, real B2 targets,
the real projection read out of the VoiceChat checkpoint, the real frozen LM in
NF4, and the real cached prompt prefixes. Over six German clips:

- token cross-entropy **0.60–1.14 nats**, all finite;
- gradient norm at the projection **2.0–4.8**, all finite, so supervision does
  reach the 4,592,000 trainable parameters through the checkpointed Mamba2 and
  attention backward;
- `assert_frozen()` holds afterwards — no language-model parameter carries a
  gradient;
- **peak 7.96 GiB** allocated (8.68 GiB reserved), against §10's ~8 GB NF4
  estimate;
- **~0.55 s per example** forward and backward, after a 65 s model load.

That last number is the one that decides the plan's feasibility. At two epochs
over the 100% budget, both prompt conditions and the teacher-gate retention,
one arm's largest fit is roughly three hours, so all twelve fits — four arms at
25/50/100% — are on the order of a day of GPU, which is what §10 predicted.
This measures the loop, not any arm: no projection was updated and nothing here
is a result.

`export` was exercised the same way, on a scratch fit whose "trained"
projection is the original VoiceChat one. The artifact it wrote carries
`voicechat_interface_fit` rather than the generic alignment fallback, its
`encoder.*` is byte-identical to E1's source (`5c2e4dd7…`, 636 tensors,
609,141,760 values), its four interface tensors match the fit exactly, and the
exported `proj.weight` is bit-identical to the projection that went in.

Partial teacher quality, over the 408 German clips generated under both prompts
so far: prompt A usable on 83%, prompt B on 90%, consistent with the §11 gate's
0–16% and 10–18% costs. **The paired retention rule compounds them**: requiring
both conditions to pass keeps 76% of clips, so Dataset B's effective size at
100% is well below its clip count. Worth reading again over fr and ru before
concluding anything about budget.

## The B1 audio teacher, exercised

Run on 24 SLURP clips under both prompts into a scratch directory, so nothing
in the real experiment was frozen at settings that might still change:

- the `VC_DUMP=1` frame trace parses into a per-frame timeline — median 79
  frames, the prefix length derived from the tokenized system prompt matching
  the runtime's first traced frame exactly (26 under prompt A, 29 under B);
- **83% of clips are retained** under the paired rule, prompt A usable on 83%
  and prompt B on 88%;
- **median 16.6 s per turn** (max 29.7), so the full pool of 2,033 clips under
  two prompts is on the order of **19 hours** — the longest single stage in the
  comparison, and it cannot share the card with the B2 text teacher.

One thing to decide before the full run rather than after it. Two of the 48
traces were rejected as "spontaneous function activity", and in both the cause
is a **single frame** where the function head emitted a subword echoing the
text channel — at `t=65` the text channel emits `'uff'` and the function
channel `'uffle'`, inside the word "truffle". That is not a tool call the
student would have to reproduce; it looks like the function head's ordinary
output on a non-tool turn. The guard is doing what it says, and it was left
exactly as written, but at 4% of traces it is worth asking whether a whole clip
should be dropped for one such frame, or whether the rule should be about the
function channel actually opening a call. Changing it is a supervision change
and belongs in the design record, not in a runner.

The teacher also mishears: on SLURP 10017 the original VoiceChat answers about
poaching truffles. That is not a defect here — B1's target is by definition
what the original model emits on the clip — but it is worth remembering when
reading B1 losses, which are distillation distances and not correctness.

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

- **The leftover Comparison 6 container was stopped, and is not running.** It
  had held 14.6 GB since that comparison finished 21 hours earlier, which was
  enough to make the fitting backward fail in `cublasCreate`; the loop only
  measured after it was stopped. Nothing else was changed and it restores with

  ```bash
  ASR_MODEL=regmean-plus-plus docker compose \
    --env-file .cache/experiments/voice-assistant-pilot-v2/runtime.env \
    -f /tmp/nemotron-voicechat-main-229dc0e/docker-compose.yml up -d voicechat
  ```

  B1 target generation needs it back with `ASR_MODEL` naming the **original**
  perception encoder rather than a candidate.
- **B1 and B2 cannot share the card.** B1 runs `voicechat-cli` by `docker exec`
  inside a container whose own bridge server already holds ~14.6 GB, and the
  B2 text teacher holds 6 GB, so the three do not fit in 24 GB together. Run
  the pools one after the other, and stop the text teacher before fitting.
- **B1 is the long pole, not B2.** 2,033 clips under two prompts is 4,066
  runtime turns, each streaming its audio and then decoding, against B2's
  ~14 hours. Budget for it before starting. Its `VC_DUMP=1` frame trace, which
  `parse_audio_trace` needs and which `asr_align/gating.py` injects, is present
  in the pinned runtime (`voicechat-cli.cpp` emits the `DUMP t=… txt=… fn=…`
  line the parser matches), so the path is supported but unexercised.
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
