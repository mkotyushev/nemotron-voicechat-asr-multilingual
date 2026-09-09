# Comparison 7: E1 fitting results and continuation

Status: **v1 is a negative diagnostic run, and its cause is now measured**. E1
at the 100% budget finished fitting and passed its internal English CE gate,
but produced **no assistant response on any of the 12 deployment pilot clips**.

The fusion-weight defect found first — the fitting code used function-channel
weight 1 where the checkpoint and runtime require 2 — is real and is corrected,
but it is **not** why the pilot was silent, and correcting it alone would not
have fixed the pilot. The measured cause is a **teacher/serving mismatch in the
audio tail**: the B1 targets were recorded through the runtime's whole-wav
`run_turn` path, which drops the audio channel to an exact zero vector once the
wav is spent, while the Realtime bridge the pilot uses feeds *encoded PCM
silence* instead and never produces that zero. The fit learned to open its turn
on a cue deployment never sends. Details and the four-cell measurement are in
"Why the pilot was silent" below.

**Refitting E1 under the corrected fusion graph is therefore not yet
authorized**: the objective, the B1 supervision and the English gate all still
carry the defect, and a corrected refit would be expected to reproduce the
silence. The prepared experiment
`.cache/experiments/comparison-7-interface-v2-fusion/` reuses frozen data,
targets and encoder caches and holds a corrected fusion graph, but its targets
are the affected ones. All remaining budgets, the fitting-precision comparison,
MASSIVE scoring and the full comparison remain pending. No deployment claim is
made.

The experiment is `.cache/experiments/comparison-7-interface-v1/`, shared with
the main checkout through this worktree's `.cache` symlink. The fit was run
from `worktree-comparison-7-interface-fitting` at `ab68272`.

## Completion and recovery check

There was no failed finalization to recover. The final optimizer update was
logged at **11:19:09 host time**; validation then ran without progress logging,
and `fits/E1-100-nf4/result.json` was written at **11:23**. The process exited
and released the GPU. The absence of another training-log line did not mean
that validation or saving had failed.

The saved `projection.safetensors` matches the last optimizer checkpoint
exactly. The checkpoint records epoch index 1, cursor 10,700 and step 2,676,
matching two completed epochs and the frozen result. Its tensors are finite,
and both the projection file hash and the candidate-provenance digest validate.

Reusable audit: `analysis/E1-100-nf4/completion_audit.json` within the experiment.
It pins the result, provenance, checkpoint and original training log and records
the checks and per-epoch summaries. The CLI now logs resume position, final
validation progress and the completed result path; these messages do not change
the objective or the recorded fit.

## Fit and held-out trace loss

Only the 1024→4480 projection and its bias were optimized: **4,592,000
parameters**. The input cache records 636 frozen F32 encoder tensors and uses
PT_ML's runtime configuration. The teachers remain B1's original VoiceChat
audio path and B2's native-transcript chat path. One projection was trained
under both frozen system prompts.

AdamW: learning rate 0.0003, no weight decay, gradient accumulation 8, gradient
norm clipping at 1, seed 0, two epochs. There were **5,350 retained recordings**
under two prompts: 10,700 clip/prompt examples per epoch, **21,400 example
presentations** overall. The 1,174 validation examples represent 587 recordings
under both prompts.

| Epoch | Optimizer updates | Mean logged batch loss | Last 100 updates' mean loss | Median gradient norm |
|---|---:|---:|---:|---:|
| 1 | 1,338 | 0.270650 | 0.226992 | 0.468715 |
| 2 | 1,338 | 0.199771 | 0.192912 | 0.352125 |

Training updates span about 3 h 35 min. These are means of the rounded,
pool-weighted batch losses in the log, not a separately recomputed training
evaluation. They are not directly comparable to unweighted validation cells.

Validation cross-entropy is in nats, lower is better. N is the number of
recordings per condition; the same recordings occur under A and B.

| Pool / input | N per condition | A: English output | B: input-language output |
|---|---:|---:|---:|
| B1 / English | 189 | 0.190823 | 0.188458 |
| B2 / German | 123 | 0.243112 | 0.299412 |
| B2 / French | 131 | 0.276148 | 0.384231 |
| B2 / Russian | 144 | 0.273009 | 0.358167 |

The two English conditions have similar final trace losses. Foreign condition
B has higher loss than condition A in each language. The conditions have
different target sequences, so these differences do not by themselves establish
better content or language compliance under A. Both are still teacher-forced
scores, including listening and padding tokens. No free-generation or
tool-calling success has been measured for this fit. E1 uses the English
encoder; any foreign-language result it eventually achieves will be the control
against which E2–E4 must be judged.

The design's statement that B1 has zero loss at initialization should not be
read literally for this implementation: cross-entropy against hard generated
tokens need not be zero even with the same model. Here the teacher also uses
Q8_0 while the fitting LM uses NF4. The gate compares the fitted and original
projections under the same NF4 scoring path instead of assuming a zero baseline.

## v1 English gate: passed internally, invalid as a deployment-graph check

The existing 0.05-nat per-cell threshold was applied without modification to
189 held-out English recordings under each prompt:

| Condition | Original projection CE | Fitted projection CE | Fitted minus original | Verdict |
|---|---:|---:|---:|---|
| A: English output | 0.299744 | 0.190823 | **−0.108921** | pass |
| B: input-language output | 0.300608 | 0.188458 | **−0.112149** | pass |

Mean CE improves from 0.300176 to 0.189641 (−0.110535 nats, about 36.8%). The
recomputed fitted scores exactly reproduce the fit's saved English validation
cells. Source hashing and the 636-tensor encoder digest check passed. The gate
finished at 11:42:25 host time; it scores 378 clip/prompt examples, not 378
distinct recordings.

Artifact: `E1_english_gate.json`; manifest digest
`60175431bad0af640779b89be2dd123e3d003ba0c6c2418865006ed6ee266e13`.
This was initially taken as permission to continue to export and evaluation.
The later graph audit invalidated its use to authorize further arms. It does
not establish the English speech-to-action requirement or a multilingual benefit.

## Export and shared evaluation

`artifacts/E1-100-nf4/` contains the full F32 encoder, fitted projection,
featurizer, PT_ML configuration and fit provenance. Reloading confirms all 636
encoder tensors remain byte-identical to E1's source; the four attached
projection/featurizer tensors exactly match their inputs. Conversion used the
existing pinned reader `/srv/bulk/ai/build/llama-voicechat.cpp`; the unavailable
default `.cache/llama-voicechat.cpp` was not bootstrapped or replaced.

The Q8 artifact is `deployment/E1-100-nf4-Q8_0.gguf`, 724,852,384 bytes, SHA-256
`14d66983e0d2b9af4ec9872faa60c13e43897a49926fb9ffe0a21ac8dbd196f9`.
A byte-identical copy is installed as
`mmproj-asr-comparison7-E1-100-nf4-Q8_0.gguf` in the runtime model directory.

The new `interface_fitting.py evaluate` stage reuses the common collection
passes and evaluator, verifies the actual Q8 artifact against the rounding
model, and saves both precision-stage results and reusable embeddings against
the exact frozen Comparison 1 arrays. Encoder/projection metrics do not
consume a system prompt; output-language conditions still need separate
deployment tables. E1's shared evaluation and runtime parity check completed.

| Shared metric | Pre-quantization | Post-Q8 |
|---|---:|---:|
| English VoiceChat-space R² | −2.254609 | −2.255960 |
| English VoiceChat-space cosine | 0.542347 | 0.542008 |
| Historical FLEURS top-1, de | 0.211268 | 0.225352 |
| Historical FLEURS top-1, fr | 0.210526 | 0.210526 |
| Historical FLEURS top-1, ru | 0.131034 | 0.131034 |
| Intrinsic cross-lingual top-1, de | 0.232394 | 0.225352 |
| Intrinsic cross-lingual top-1, fr | 0.187970 | 0.187970 |
| Intrinsic cross-lingual top-1, ru | 0.131034 | 0.124138 |

English evaluation has 21,660 frames from 285 frozen validation clips; FLEURS
has N=142/133/145 for de/fr/ru. Full top-5, MRR, ranks, counts and paired
confidence intervals are in the common result records. Pre-quantization
English R² has a paired difference versus PT_ML of approximately −1.5538,
95% CI [−1.5799, −1.5261]; the cosine difference CI is [+0.2401, +0.2431].
The projected-frame mean norm is 4.4407 against FT_EN's 2.3700, about 1.87×.
Thus the poor R² is accompanied by substantial displacement and norm growth,
despite the improved teacher-forced loss. Q8 changes R² by only −0.001351.
FLEURS did not select or modify the graph fix, which follows directly from the
runtime and original configuration.

All 640 tensors (encoder plus interface/featurizer) in the actual Q8 artifact
match the simulated rounding exactly. Runtime parity passes at the original
0.05-sigma tolerance: the largest checked difference is 0.0222 sigma.
The parity artifact is `analysis/E1-100-nf4/runtime_parity.json`, digest
`3819dbca1bf37056572ae02ecaa0296a6edb8ceb964922ad805c7fe1efe77698`.
Shared results and reusable pre/post embeddings are in
`evaluations/E1-100-nf4/`, whose run manifest digest is
`83d77a30eace7bba9525a203ef360a7409f1c03f71aeddcd25669a7eac376b68`.

## v1 deployment pilot: failed English control

The original frozen pilot was reused with its existing prompt, runtime,
environment and 30-second response budget. This is the historical development
pilot, not completion of the new condition A/B deployment evaluation.

| Candidate | English exact calls | Russian exact calls | Responded EN / RU |
|---|---:|---:|---:|
| Original FT_EN control | 4/6 | 0/6 | 6/6 / 6/6 |
| Comparison 3 | 1/6 | 0/6 | 1/6 / 0/6 |
| Comparison 6 RegMean++ | 3/6 | 0/6 | 4/6 / 0/6 |
| v1 E1 fitted interface | **0/6** | **0/6** | **0/6 / 0/6** |

Every E1 record is `no_response`, with no tool call, assistant text or recorded
transport error. Discovery verified `comparison7-E1-100-nf4` and a ready
backend before the run. The paired Russian-minus-English accuracy difference
is zero because both languages failed; that is not evidence of parity of
capability.

Records: `voice-assistant-pilot-v2/interface-E1-100-nf4-post-q8/`.
The six-row table was made by `build_comparison()` in the new directory
`voice-assistant-pilot-v2/analysis-through-comparison-7-E1-v1/`. The original
control's frozen raw trace was rescored into that directory with the existing
1.2 scorer, because its older saved result used schema 1.1. Existing results
and tables were not overwritten. A fresh FT_EN control recheck is also running.

## Fitting-graph defect and corrected experiment

The deployed `voicechat-cli.cpp:step` sums
`1*text_embedding + 2*function_embedding + 1*audio_embedding`. The coefficients
come from the original `model.stt.model.duplex_*_channel_weight` settings and
are reproduced in the function-head GGUF metadata. They apply to PAD tokens
as well as function-call tokens, and apply during system-prompt conditioning.

The v1 `duplex_inputs` and `cache_prompt` instead added every channel at unit
weight. Both projections in its gate were evaluated through that same wrong
graph, so a passing gate could not detect the mismatch. Encoder parity also
could not detect it: that check ends at the projected encoder embeddings,
before duplex fusion into the LM.

The fix reads and pins the original fusion configuration, uses one F32 fusion
operation for both paths, and records graph version
`voicechat-duplex-fusion-v2` in candidate provenance. Legacy experiment
manifests, optimizer checkpoints and gradient calibrations cannot silently
resume under the new graph. No served weight, data budget, teacher, objective,
learning rate or epoch count is changed. The decision is recorded in
`AGENTS.md`.

This is a confirmed implementation mismatch, not yet proof that it is the only
cause of the silent deployment. A diagnostic projection analysis found relative
weight movement of 0.7256 L2 and only 0.0572 cosine between the mean projection
shift and the missing PAD embedding, so a simple constant-offset compensation
story is not supported. That analysis is frozen in
`analysis/E1-100-nf4/fusion_offset_diagnostic.json`. A direct fusion ablation
on held-out English clips is the next check before the corrected E1 fit.

The v2 experiment manifest digest is
`1ae8bf920c477beb86dc0a3a784a899e2254694c064c28d58a55189ff66423f0`;
`reused_inputs.json` has digest
`b4f61a3c3ce1b53f85f24b9ecc2ce7380380479b1266c0be61377f028d2aaf3b`.
Each of the four reused caches has 7,043 clips and matches the new experiment's
source, Dataset B digest and runtime configuration. Changed fusion policy is
rejected by the frozen writer. Re-preparing v2 with identical inputs is refused
by the frozen manifest writer, which is the intended behaviour.

## The fusion ablation: the graph fix is real, and is not the cause

The ablation the previous section called for has run, on 24 held-out English
recordings under both prompts, scoring the original and the v1 fitted
projection under both fusion graphs with nothing else changed. Cross-entropy
in nats, lower is better:

| Scoring graph | Original projection A / B | v1 fitted projection A / B |
|---|---:|---:|
| Corrected, function weight 2 (deployment) | 0.270055 / 0.279293 | 0.179641 / 0.189491 |
| Unit weight, function weight 1 (v1's graph) | 0.308204 / 0.314363 | 0.182876 / 0.194174 |

The correction mostly helps the **original** projection, which is what one
should expect: weight 2 is the graph that checkpoint was trained for. It barely
moves the fitted projection, and under the corrected deployment graph the v1
fit still beats the original by about 0.090 nats in both cells. So the fusion
defect did not manufacture the v1 gate result, and repairing it does not by
itself change any conclusion about the fit. Artifact:
`analysis/fusion_sanity.json`, digest
`e1e5fbf3b3ede85c904695be5759042dddc724296217127baa00e465ce26cf5a`.

## Where the fitted projection's cross-entropy gain actually lives

The objective is a uniform mean over every frame of the teacher timeline
(`interface_fit.token_loss`: `total / len(labels)`), and a B1 timeline is
63.7% PAD. Splitting the held-out English cross-entropy by target class, under
the corrected fusion, over 48 recordings under both prompts:

| Target class | Frames | Original | Fitted | Fitted − original | Share of the total gain |
|---|---:|---:|---:|---:|---:|
| all | 6,835 | 0.2885 | 0.2049 | −0.0836 | 100% |
| pad (listening) | 4,062 | 0.0612 | 0.0070 | −0.0542 | 39% |
| onset | 192 | 1.9393 | 0.0316 | **−1.9077** | **64%** |
| continuation (the reply) | 2,581 | 0.5235 | 0.5293 | **+0.0058** | −3% |

Mean predicted P(PAD) on PAD frames rises from 0.9672 to 0.9956, and on PAD
frames while the command is still playing from 0.9667 to **0.9999**.

The onset frames carry two thirds of the gain and are worth nothing in
deployment. Every one of the 378 retained B1 validation traces has exactly two,
and they are always the same two: an EOS at **absolute frame 9** in all 378
traces, where the model closes the system prompt's turn, and a BOS at exactly
the **first frame past the audio** in all 378 traces. The second is not a
decision the model made — `VC_FORCE_BOS=1` forced it, and the runtime forces it
again at serving time whenever that code path is used. So the fit's measured
improvement is 64% on a positional constant and a forced token, 39% on being
more confidently silent, and slightly negative on the reply content, which is
the only part the pilot scores. Artifact:
`analysis/pad_mass_diagnostic.json`, digest
`a29bfae2c1d464c8f006c65a5936859987339afe6d5145b5670afa43c2acddf1`.

This is also why the English gate passed. It is a uniform frame mean over the
same timelines, so it cannot separate a projection that answers better from one
that is merely more certain about padding and forced tokens.

## Why the pilot was silent

The v1 pilot records show no transport error and no `response.created` event at
all: only `session.created`, `session.updated`, `speech_started` and
`speech_stopped`, then the 30 s budget expiring. The control emits the full
`response.*` sequence and first text at about 1.4 s past the audio. The Q8
artifact is not implicated — runtime parity passes at 0.0222 sigma and the
encoder produces 85 frames for the 6.68 s parity clip, as expected.

Two runtime paths matter, and the experiment used one for supervision and the
other for evaluation.

- `vc_session::run_turn` is the legacy whole-wav path. `PerceptionPathEngine`
  drives it with `{"cmd":"turn","audio":...}`, and it produced **every B1
  teacher trace**. It honours `VC_NO_BARGE` and `VC_FORCE_BOS`, and once the wav
  is spent it passes `a = nullptr`, so the audio channel becomes an **exact zero
  vector**. `interface_fit.duplex_inputs` reproduces exactly this convention:
  `projected[indices.clamp_min(0)] * (indices >= 0)`.
- `vc_session::duplex_step` is the path the Realtime bridge drives with
  `duplex_start` / `audio_frame`, and it is what the **deployment pilot** used.
  It clears `hold_bos` and `want_bos` on every frame, so it honours neither
  `VC_NO_BARGE` nor `VC_FORCE_BOS`; the model must emit BOS itself. And
  `bridge/server.py::_audio_loop` advances the model at 12.5 Hz "using silence
  on an input underrun", feeding PCM silence through the ASR encoder once the
  client stops sending. Encoded silence is not a zero vector.

`runtime.env` marks the distinction in a comment directly above the three
variables: "Legacy whole-wav turn mode. The Realtime bridge uses the duplex
stream path."

Free-running greedy decode in the fitting harness, on 12 held-out English
recordings under both prompts, under the bridge's duplex rules (nothing forces
BOS), with the audio tail supplied both ways:

| Audio tail after the command | Original projection | v1 fitted projection |
|---|---:|---:|
| Exact zero — what the teacher traces and the fitting graph use | 21/24 responded | 22/24 responded |
| Encoded silence — what the bridge actually feeds | **24/24 responded** | **0/24, never opened a turn** |

The original projection is indifferent to which tail it receives. The fitted
projection answers fluently on the tail it was trained against and **never
opens a turn** on the tail deployment sends, running to the 12 s extension cap
in silence on all 24. That reproduces the pilot's 0/12 `no_response` offline,
from frozen artifacts, and identifies the cause: the fit learned to open its
turn on an exact-zero audio cue that the deployed path never produces, and the
supervision taught it that the opening BOS would be handed to it anyway.

Artifact: `analysis/silence_tail_diagnostic.json`, digest
`68becb3f5fc9b8e27bfe0afb84f597fe44d0bef8f1cabd23c9b6c442c66eb35e`. A
companion run under `run_turn` rules, where BOS is forced and the tail is zero,
has both projections responding 48/48 with fluent English:
`analysis/free_running_diagnostic.json`, digest
`b7a12d4c32dd8f1f1add2d037f34b43930645f496e77b37f5f45729f676b95de`. Both are
behavioural proxies in the NF4 fitting harness, not the deployed Q8 runtime
with its TTS channel and VAD epochs.

## Duplex teacher subset: the frozen traces cannot be repaired

A 12-clip subset (24 clip/prompt targets, the same held-out English clips every
other Comparison 7 diagnostic uses) was regenerated through the runtime's duplex
path — same container, same original Q8 teacher, same mmproj, same two system
prompts, driven over `duplex_start` / `audio_frame` instead of a whole-wav
`turn`. It is compared against the frozen `run_turn` trace for the same clips
with its two harness artifacts undone on paper: the forced BOS marked as the
runtime's rather than the teacher's, and the `-1` audio indices understood as
encoded silence rather than a zero embedding.

| | Regenerated (duplex) | Repaired (run_turn) |
|---|---:|---:|
| Opened a turn | 24/24 | forced, by construction |
| BOS frame relative to the command end | −33 to +5, 8/24 before it ends | always exactly 0 |
| Reply text identical between the two | 7/24 | — |
| EOS at frame 9 | 22/24 | 24/24 |
| Mean spoken tokens | 19.54 | 18.46 |

**The repair is not a faithful stand-in, and the frozen B1 traces cannot be
patched into duplex supervision.** Three findings drive that:

- **Onset timing is not recoverable.** In `run_turn` the BOS sits at offset 0 in
  every trace because `VC_FORCE_BOS` put it there. The duplex teacher opens
  anywhere from 33 frames before the command ends to 5 frames after. The
  whole-wav trace carries no information about when the teacher would have
  chosen to speak, so a repair can only mark the frame as forced — it cannot
  supply the timing that a duplex student has to learn.
- **Barge-in cannot be represented at all.** 8 of 24 duplex turns open while the
  command is still playing, which `VC_NO_BARGE` made impossible in the frozen
  run. That is a behaviour class absent from the existing targets.
- **The replies themselves differ**, in 17 of 24 cases, sometimes in meaning
  rather than wording. On `B1/en/2312` the duplex teacher says "I am unable to
  control your lights, but I can…" where the frozen trace says "The light is now
  down to seven." On `B1/en/9606` it asks "Which programs would you like to
  play?" against the frozen "I cannot play programs, but I can help you with…".
  So the two teachers are not the same supervision with different framing.

Two things the comparison also settles. The **EOS at frame 9 is real model
behaviour**, not a `run_turn` artifact: it appears in 22 of 24 duplex traces
too, where the model closes the system prompt's turn. It remains a positional
constant that is trivially predictable and should not be allowed to dominate a
loss, but it is not something regeneration removes. And the **original teacher
has no difficulty opening its own turn** — 24/24 — so a duplex teacher is
viable; nothing about the checkpoint requires the forced BOS.

One caution for whoever regenerates the full pool. Early barge-in is genuine
duplex behaviour but is not always *good* supervision: on `B1/en/10925` the
teacher opens 33 frames (2.6 s) before the command ends and answers "What would
you like to do?" instead of addressing the request, having not yet heard it.
A duplex regeneration needs a usability filter at least as strict as the
existing one, and probably an explicit check that the turn opened after enough
of the command to answer it.

Artifacts: the regenerated targets are `targets-duplex/` under the v2
experiment, index digest
`0a1bba9e7bf3691d5ab1693c8ea3deba0d00721d4aff807e2e68f17392df49b6`, with the
full per-frame `VC_DUMP` trace kept per target; the comparison is
`analysis/teacher_shape_comparison.json`, digest
`13d0a59633a2bdabadad85147d81b9af581e699d3f4561579ac70a772f67d2f5`. The driver
is `.cache/experiments/comparison-7-duplex-teacher.py`. This is a 12-clip
subset for shape comparison, not a teacher pool: no fit may consume it.

## The duplex regeneration recipe, and what it retains

Decided: B1 is regenerated through the duplex path, and **barge-in is not
supervised**. Opening a turn over the user is real FT_EN behaviour and stays in
the frozen LM's own weights; teaching it here would train the student to answer
before it has heard the request, and the frozen `run_turn` pool could not
represent it anyway. The recipe is otherwise the frozen B1 recipe — same
container, teacher, mmproj and prompts.

Added to it:

- **The command boundary comes from the encoder, not the block count.** The
  streaming encoder has its own startup latency and the last block is zero
  padded, so `frames` on each `duplex_frame` acknowledgement is summed instead.
  Measuring it as blocks-sent was wrong by about a frame: `features.frames_out`
  is uniformly `blocks + 1` on these clips.
- **A 4-frame onset tolerance.** A turn opening one or two frames before the
  encoder has formally consumed the command is boundary slop, not barge-in, and
  the replies show it: `B1/en/9606` at −2 answers "Which programs would you like
  to play?" and `B1/en/15269` at −1 answers "I cannot post to your Facebook…" —
  complete and on topic.
- **A 0.64 s silence lead-in** before the command, as a live microphone stream
  has and as the pilot's own clips do (320 ms). At n=48 it moved retention from
  75% to 81%, which is inside the noise; it is kept because it matches the
  deployment condition, not because the difference is significant.
- **The existing usability filter unchanged**, and a rejection if the turn takes
  more than 50 frames (4 s) to open.

Retention over 48 held-out clips under both prompts, 96 targets:

| | Duplex, this recipe | Frozen B1 (`run_turn`) |
|---|---:|---:|
| Per-target retention | 0.854 | 0.94 / 0.95 per prompt |
| Paired retention (both prompts) | **0.833** (40/48 clips) | 0.93 |
| Opened a turn | 96/96 | forced |
| Onset past the command, retained | median 4 frames (0.32 s), −4 to +9 | 0 by construction |

Rejections are 10 barge-ins, 4 unusable replies and 2 with no identified
language. **Barge-in is deterministic per clip, not sampling noise**: every
barged-in clip fails under both prompts at nearly the same onset (−10/−10,
−33/−34, −6/−6, −7/−7, −5/−5), so it is a property of the audio — most likely a
mid-utterance pause the teacher reads as the end of the turn — and the filter
that removes it is stable rather than arbitrary.

The onset distribution is the encouraging part: 33 of 82 retained targets open
at exactly +4 frames and 63 of them within +3 to +5, so the teacher's natural
turn-taking is consistent enough to be worth supervising. That is the behaviour
the forced BOS was destroying.

So the pool costs about 17% of its clips against the frozen one, and paired
retention of 0.833 over B1's 2,033 clips projects to roughly 1,690 retained
clips against the frozen pool's ~1,890. At the measured ~16 s per turn, 4,066
turns is about 18 hours, matching what B1 cost.

Artifacts: `targets-duplex-lead0/`, `targets-duplex-lead8/` and
`targets-duplex-check48/` under the v2 experiment, each with its own
`provenance.json`, per-target `VC_DUMP` trace and `index.json` carrying the
retention summary. The generator is
`.cache/experiments/comparison-7-duplex-subset-check.py`. These are subset
checks sizing the full run; no fit may consume them.

## The full duplex pool, generated

2,033 clips under both prompts, 4,066 targets, complete. Generated 2026-09-09
23:00 to 2026-09-10 03:57 at about 8 clips a minute, against the 17.7 hours the
v1 pool cost.

| | Duplex, full pool | 48-clip pilot | Frozen B1 (`run_turn`) |
|---|---:|---:|---:|
| Per-target retention | 0.764 | 0.854 | 0.94 / 0.95 per prompt |
| Paired retention | **0.664** (1,350 clips) | 0.833 | 0.93 |
| Onset past the command, retained | median 4, −4 to +19 | median 4, −4 to +9 | 0 by construction |
| Frame accounting exact | **4,066 / 4,066** | — | — |

Retention by cell, with no cell close to empty: `train/A` 68.2%, `train/B`
84.9%, `validation/A` 69.0%, `validation/B` 80.3%. The prompt asymmetry is
consistent across splits and is barge-in: the English-only prompt draws more of
it. Budgets survive the filter — the 25% budget keeps 323 of the 457 B1 clips
it draws, the 50% 619 of 915, the 100% 1,215 of 1,830 — and the held-out split
keeps 303 paired targets, which is what the gate scores.

Rejections are 740 barge-ins, 232 unusable replies, 73 wrong or unidentified
language, 66 turns that never opened or said nothing, and 2 tool calls. All are
deliberate: barge-in and tool calls are FT_EN's own behaviour and stay in the
frozen LM's weights.

The pilot's 0.854 was measured **without** the trace-parse gate, which is why
the full pool reads lower rather than because anything degraded; see the run
log's trace-slicing entry.

### Changing the turn path changed the teacher's answers, not just their timing

Over the 3,046 clip/prompt pairs usable under both paths, the duplex reply and
the v1 `run_turn` reply agree at a median token F1 of **0.600**, and only 21.8%
are identical. At temperature 0 the decode is deterministic, so this is not
sampling: the context genuinely differs, because the model picks its own BOS
position instead of having one forced and hears encoded silence instead of a
zero vector after the command.

This settles a question the subset check could not. Regenerating B1 was not a
cleanup of slightly noisy labels — the v1 targets were *a different target*.
Roughly four fifths of the pool would have taught the student a reply the
teacher does not give on the path deployment runs.

### The gate's control works, checked before the refit was started

Cheap to run and expensive to get wrong, so it was run first: FT_EN's own
untouched projection, free running on held-out English under all three
deployment conditions, opened **12 of 12** turns with fluent on-topic replies
at onsets of −5 to +4 frames, median +3 — against the duplex teacher's own
median of +4.

That validates the whole chain end to end rather than the harness alone. The
model produces coherent speech while its audio channel is indexing the cache's
encoded silence, which is the supervision the refit consumes; the frame
alignment at offset 1 puts the command where the model expects it; and the
control is calibrated, so `duplex_gate_verdict` can tell a silent fit from a
sound one instead of returning a meaningless pass. Free running costs about
4.7 s an example, so the full gate over 303 held-out targets under two
projections is roughly 45 minutes.

## Two preconditions the refit had beyond the teacher, both now met

Regenerating B1 fixed the supervision but not the graph that consumes it.
Both of these are implemented and unit tested; neither has run on the GPU yet.

1. **The training graph fed an exact zero audio tail.**
   `interface_fit.duplex_inputs` computed
   `projected[indices.clamp_min(0)] * (indices >= 0)`, so every post-command
   frame was a zero vector. A duplex teacher hears *encoded silence* there, and
   so does deployment. Left unchanged, the refit would have presented zeros
   where its own teacher heard silence — the same train/serve gap with better
   labels.

   The new `cache-duplex` stage re-runs each clip's encoder over the waveform
   the bridge actually streams — lead silence, the command zero padded to a
   whole 80 ms block, then a silence tail — and both timelines index it as a
   contiguous window. `duplex_inputs` now **rejects** a timeline containing a
   `-1` rather than zeroing it, so v1 supervision is not fittable by accident.
   The tail is 213 frames, taken from the pools themselves: B1's timelines
   reach 150 and B2's longest reply needs 213. There is no per-clip trimming
   and no shared steady-state frame, because silence embeddings do not
   converge — still ~0.6 from the steady state at k=128, and cosine 0.85–0.995
   between clips — so a shared tail would be a different tail. That costs
   7.7 GB an arm, which is why the tensors live on `/srv/bulk` while the
   sidecars and index stay with the experiment and are still hashed.
2. **The gate must run free, on the deployment tail, and score onset.** A
   free-running check alone would not have caught v1: with the training-time
   zero tail both projections answered 48/48. It has to reproduce all three
   deployment conditions — nothing forces BOS, no barge-in suppression, and an
   encoded-silence tail — and record whether the turn opened and how many frames
   past the command, not merely whether tokens appeared. The untouched FT_EN
   projection is the negative control and passes at 24/24.

   `interface_fit.duplex_free_run` reproduces all three and reports the onset;
   `duplex_gate_verdict` fails the fit if its open rate falls more than five
   points below the untouched projection's in any prompt cell, and fails the
   *whole gate* if that control does not itself open every held-out turn —
   an uncalibrated harness must not be able to return a pass. `english_gate`
   now runs both and requires both. A turn that has not opened 50 frames past
   the command stops there rather than decoding the remaining ~200 frames of
   silence, since on this gate the silent runs are the common case.

## What has to be decided before any arm is refit

These are changes to the objective and the blocking check defined in
`REGMEAN_INTERFACE_DESIGN.md`, so they need a recorded decision rather than an
implementation choice:

1. **B1 supervision.** The targets encode a forced turn opening and an
   exact-zero audio tail. The subset comparison above settles how to fix it:
   **regenerate through the duplex path**. Repairing the frozen traces cannot
   work, because onset timing was destroyed by the forced BOS, barge-in was
   suppressed outright, and 17 of 24 replies differ anyway. Regeneration costs
   what B1 cost — 4,066 runtime turns, about 17.7 hours — so the budget and the
   usability filter are worth settling before it starts.
2. **The objective — less is wrong with it than first appeared.** The proposal
   to keep only N=10 frames of trailing pad is **already satisfied**: measured
   over all 378 retained B1 validation traces, truncating the tail to 10 drops
   0.0 frames on average, because `run_turn` ends a turn on a 10-frame pad
   streak and `text_target_timeline` appends exactly `[PAD] * 10`. The PAD that
   remains is 47.2% during the command and 13.2% after it, not tail padding.
   Since the forced BOS carried 64% of the v1 gain and duplex regeneration
   removes it, the uniform mean may be defensible once the teacher is fixed.
   The one class still worth masking is the **EOS at frame 9**, where the model
   closes the system prompt's turn: it is a positional constant in every trace,
   and the reference has direct precedent for masking a conditioning region.
   The listening PAD should *not* be masked away — remaining silent while the
   user speaks is behaviour the reference preserves and measures.
3. **The English gate.** As the same uniform mean it certified −0.11 nats for a
   projection that cannot open a turn in deployment. It needs the free-running
   duplex check described above alongside the teacher-forced cross-entropy, and
   that check must score turn opening explicitly.

### What the reference says, and does not

The repository has no STT-side duplex training recipe. `papers/` holds
RegMean++ (model merging) and VoiceChat-TTS, which is the *speech decoder*, not
the duplex text LLM being distilled here. VoiceChat-TTS does supply three
relevant precedents, and they are consistent with the decisions above:

- The text channel is "right-padded with special padding IDs … until they match
  the total temporal length of the current conversational turn", so the padded
  timeline shape is the intended design rather than an artifact.
- "The loss over this prompt region is masked so that the model uses the prompt
  as conditioning context rather than learning to reconstruct it" — masking a
  conditioning region is what the reference does.
- Silence while the user speaks is treated as behaviour to preserve and is
  measured: intelligible speech during PAD-designated intervals is scored as
  ASR insertion error. So the listening PAD is signal, not noise.

How the STT side aggregated its own duplex loss is not recorded here and was
not found; the decisions above are reasoned from the runtime and from these
precedents, not from the original recipe.

The fusion-weight correction and its regression tests are sound and are kept.

## Frozen budgets and provenance

The paired teacher filter retained 11,874 of 14,086 clip/prompt entries across
train and validation. The nested training budgets, after that filter, are:

| Dataset B budget | English | German | French | Russian | Total clip/prompt examples |
|---|---:|---:|---:|---:|---:|
| 25% | 868 | 584 | 610 | 642 | 2,704 |
| 50% | 1,714 | 1,176 | 1,196 | 1,282 | 5,368 |
| 100% | 3,418 | 2,334 | 2,396 | 2,552 | 10,700 |

Every arm inherits the frozen E1 calibration: B1/B2 initial median gradient
norms 2.932548 / 5.073527, giving weights **1.267419 / 0.732581**.

| Record | Digest |
|---|---|
| Experiment manifest | `6432d20a11a4953b70f1af687c8f0d10af2952626901afb03b386d886af6f47d` |
| Dataset B manifest | `f8595541d97b4200583ee8e69a309f916f5258ad106e4969f0366dbf3ed81078` |
| Teacher index manifest | `bddc46e9db4613221b8bc6478d9b4c6b5765970e40bd20a706a0d5285c6edc92` |
| Loss calibration manifest | `4d665e586525d67f1920f0ce59be56522574bebd777ca7a5ff898dbccb1724f2` |
| E1 encoder tensor-byte digest | `5c2e4dd741de22fb2f0ed4066e0f3371c80e82d765e2aa10330577c04a30be6e` |
| Fitted projection file SHA-256 | `117f2c9ceac9a8a508a1d768aec880601626b83f021f8aeb27c3031da0260250` |
| Candidate provenance digest | `97449f2fdefa2c11a33d9c953e24e73eb9e6ddd91db74bcb40f483f9c60f597f` |
| Fit result manifest | `3064808a891b34ffc4fbfcf17efb6fda3b9217bfa6efb95db00104e6ad43b547` |
| Completion audit manifest | `481708bd008d67e48235a1244c3aeddc245246746a5f9dfe72f82186078bcd5a` |

Manifest digests are the repository's canonical payload hashes; file hashes
are identified explicitly. The completion audit also records the corresponding
file hashes for its inputs. This is a **pre-quantization projection fit against
an NF4 LM**, not a post-Q8 evaluation.

## Commands and checks

```bash
.venv-align/bin/python interface_fitting.py fit --arm E1 --budget 100 \
  --precision nf4 --output .cache/experiments/comparison-7-interface-v1

.venv-align/bin/python interface_fitting.py gate --budget 100 --precision nf4 \
  --output .cache/experiments/comparison-7-interface-v1
```

The fitting command is recorded here for reproduction in a new experiment; the
completed frozen fit correctly refuses an in-place rerun. The gate uses 189
held-out B1 recordings per prompt and must not exceed the original projection
by more than 0.05 nats in either cell before E2–E4 can run.

Compilation, all **138 unit tests**, and `git diff --check` pass. New regression
tests cover the original fusion configuration, weighted system prefix and audio
timeline, and rejection of legacy experiment/checkpoint reuse. The added
evaluation stage preserves the common result schema and numerical evaluation
code. Actual runtime parity and pre/post artifact checks are recorded separately
when they finish.
