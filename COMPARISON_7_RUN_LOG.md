# Comparison 7 run log: end-to-end interface fitting, and the merge ablation

**Status: v1 E1 is a negative diagnostic run. It passed its internal CE gate
but failed the deployment control (no response on all 12 clips). Its fitting
graph omitted the function channel's factor of 2. The graph is corrected and
v2 is prepared; E2–E4 remain held pending a corrected E1.** This
file records what is frozen, what has been verified, and what the next stage
is, so the run can be resumed without re-deriving any of it. Measurements
are indexed in `COMPARISON_7_RESULTS.md`; the §11 gating checks
that unblocked this comparison are already in `COMPARISON_7_GATE_RESULTS.md`.

Original output: `.cache/experiments/comparison-7-interface-v1` (preserved).
Corrected output: `.cache/experiments/comparison-7-interface-v2-fusion`.
The following v1 setup and teacher history remains valid for the reused inputs;
v1's optimizer checkpoint and gradient calibration must not be reused.

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
| `targets-text` (B2) | **complete**: 10,020 targets, all present and digest-valid |
| `targets-audio` (B1) | **complete**: 4,066 targets, all present and digest-valid |
| `freeze-targets` | **frozen**: 14,086 entries, 11,874 retained, no cell emptied |
| `fit` | **E1 at 100% complete**, 2,676 updates over 10,700 examples × 2 epochs; other fits pending |
| `gate` | v1 passed internally, but is invalid for authorizing further fits after the graph audit |
| `export` | **E1 exported**; all 636 encoder tensors byte-identical and all four interface/featurizer tensors exact |
| quantization | v1 E1 Q8 artifact matches rounding exactly; runtime parity passed |
| shared evaluation | v1 E1 pre/post complete with frozen Comparison 1 arrays; R² −2.254609 / −2.255960 |
| speech-to-action | v1 E1: 0/6 English and 0/6 Russian calls, no assistant turns; original control recheck running |
| corrected fit | v2 prepared with weighted duplex fusion, targets/caches reused, fresh calibration and E1 fit pending |
| MASSIVE `test` scoring | pending |

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

## B2 teacher quality, complete

All 10,020 targets are written and digest-valid, none missing or truncated, and
the run logged no error. Usable rate per cell:

| | de | fr | ru |
|---|---:|---:|---:|
| prompt A, English only | 0.87 | 0.90 | **0.99** |
| prompt B, input language | 0.88 | 0.87 | 0.86 |

Russian's 0.99 under prompt A is the §11 gate showing through rather than a
surprise: that check found the model answering in the input language on 15% of
German and 9% of French utterances under prompt A, and Russian is the language
it is least willing to answer in, so almost nothing is discarded for replying in
the wrong one. Read together with prompt B, where ru is the weakest cell at
0.86, the two conditions are gated by opposite failures.

**The paired retention rule costs more than either rate suggests**: requiring
both conditions to pass keeps **81%** of B2 clips (4,039 of 5,010) — 0.78 de,
0.80 fr, 0.85 ru on train — so Dataset B's effective size is well below its clip
count and the 25/50/100% budgets are drawn before this gate, not after. No
language or split cell is empty, so `freeze-targets` will not trip on B2.

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

Two of the 48 traces were rejected as "spontaneous function activity", and in
both the cause is a **single frame** where the function head emitted a subword
echoing the text channel — at `t=65` the text channel emits `'uff'` and the
function channel `'uffle'`, inside the word "truffle". That is not a tool call
the student would have to reproduce; it looks like the function head's ordinary
output on a non-tool turn. At 4% of traces it is worth asking whether a whole
clip should be dropped for one such frame, or whether the rule should be about
the function channel actually opening a call.

**This was put to the decision and the guard was kept as written.** It is a
decision rather than an oversight: `usable` and `timeline` are frozen per target
file at generation time and `write_frozen` will not replace them, so revisiting
the rule later means regenerating B1 into a new directory — the stored
`runtime_trace` makes a re-parse possible in principle, but not in place.

**Over the full pool it cost far less than the sample implied**: 29 rejections
in 4,066 traces, **0.7%** rather than 4%. The 48-trace sample was simply
unlucky, which is worth remembering the next time a rate is read off a smoke
test here.

## B1 teacher quality, complete

All 4,066 targets present and digest-valid, no error in the run, 17.7 hours of
turns at a median 14.4 s (max 59.8). Timelines are median 70 frames, from 28 to
300.

| | prompt A | prompt B |
|---|---:|---:|
| usable | 0.94 | 0.95 |

**Paired retention is 0.93** (1,898 of 2,033), even across train and
validation — much healthier than B2's 0.81, because B1's teacher is being asked
only to answer its own English audio, not to hold an output language.

## The frozen supervision

`freeze-targets` wrote 14,086 entries, one per clip and prompt, of which
**11,874 (84%) are retained**, and no language/prompt cell was emptied:

| split | B1 en | B2 de | B2 fr | B2 ru |
|---|---:|---:|---:|---:|
| train | 0.93 | 0.78 | 0.80 | 0.85 |
| validation | 0.93 | 0.74 | 0.78 | 0.86 |

The **loss calibration is frozen from E1 and inherited by every arm**, which is
what makes the arms comparable: median initial projection-gradient norms are
2.93 on B1 and 5.07 on B2, so the pools enter the objective at weights 1.27 and
0.73 with mean one. That is the measured answer to the gate result that equal
term weights are not defensible — B1 and B2 losses are not on the same scale,
and this puts them there by gradient size rather than by assumption.

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

**The refit waited on B1 being regenerated, and that is now in flight.** The
fusion-weight correction is sound and its tests pass, but it is not why the
pilot was silent, and the v1 supervision the refit would have consumed still
carried the defect that was. The
measured cause is in `COMPARISON_7_RESULTS.md`, "Why the pilot was silent":
the B1 targets were recorded through the runtime's whole-wav `run_turn` path,
which forces the turn-opening BOS and drops the audio channel to an exact zero
vector after the wav; the Realtime bridge that the pilot uses forces nothing
and feeds encoded PCM silence instead. Fed the tail deployment actually sends,
the v1 fitted projection never opens a turn on 24/24 held-out English turns,
while the original projection answers 24/24.

So B1 supervision, the uniform-frame objective and the English gate each need a
recorded decision before any arm is refit. Running the command below unchanged
would be expected to reproduce the silent pilot at a cost of about 3.5 GPU
hours plus export and evaluation.

The decisions are now taken and implemented; what is left is the order below.
B1 is regenerating on the duplex path, and the two stages after it cannot start
until it finishes, because each needs the whole card.

```bash
# 1. B1 on the turn path deployment runs (in flight, ~4 h, one GPU)
.venv-align/bin/python interface_fitting.py targets-audio-duplex \
  --massive .cache/datasets/MASSIVE/1.1/data --n-gpu-layers 99 \
  --output .cache/experiments/comparison-7-interface-v2-fusion

# 2. the encoder cache over the waveform the bridge streams; ~8 GB an arm, so
#    the tensors go to /srv/bulk while the sidecars stay with the experiment
.venv-align/bin/python interface_fitting.py cache-duplex --arm E1 \
  --activation-root /srv/bulk/ai/data/comparison-7/activations-duplex \
  --output .cache/experiments/comparison-7-interface-v2-fusion

# 3. freeze against the duplex pool rather than the v1 one
.venv-align/bin/python interface_fitting.py freeze-targets --b1-directory B1-duplex \
  --output .cache/experiments/comparison-7-interface-v2-fusion

# 4. the refit, and the gate that now also runs the turn free
.venv-align/bin/python interface_fitting.py fit --arm E1 --budget 100 \
  --precision nf4 --epochs 2 --learning-rate 0.0003 --accumulate 8 --seed 0 \
  --output .cache/experiments/comparison-7-interface-v2-fusion
.venv-align/bin/python interface_fitting.py gate \
  --output .cache/experiments/comparison-7-interface-v2-fusion
```

Retention while B1 runs, and the frame-accounting check the cache depends on:

```bash
.venv-align/bin/python .cache/experiments/comparison-7-b1-duplex-progress.py
```

The three diagnostics behind that hold need no server and about 25 GPU minutes
in total. They read only frozen artifacts and write only into `analysis/`:

```bash
.venv-align/bin/python .cache/experiments/comparison-7-pad-mass-diagnostic.py
.venv-align/bin/python .cache/experiments/comparison-7-free-running-diagnostic.py duplex
.venv-align/bin/python .cache/experiments/comparison-7-silence-tail-diagnostic.py
```

The last one is the useful survivor: it free-runs a projection under the
bridge's duplex rules with an encoded-silence tail and reports whether the
model opens a turn at all. It answers in about 6 GPU minutes what previously
took a 3.5 h fit, an export and a container pilot to discover, and it would
have blocked v1.

## Two measurements the silence tail needed

**Where the streamed frame lands in the cache.** The runtime streams one 80 ms
block at a time and its encoder emits exactly one frame per block; the cache
runs the same waveform whole and `features.frames_out` returns one frame more,
so one cached frame has no counterpart in the stream. The frontend settles
which. Each of the three stride-2 convolutions pads two mel frames left and one
right, so its output `j` reads inputs `2j-2 .. 2j`; composed three times,
cached frame `j` reaches mel frame `8j` — audio through sample `1280 * j`, the
end of block `j-1`. The frame a streamer emits having just consumed block `i`
is therefore cached frame `i + 1`, and cached frame 0 is the one the stream
never emits. `interface_fit.CACHE_FRAME_OFFSET = 1`.

Two probes agree with that without pinning it further, and both are worth not
repeating. On 40 blocks of pure silence, frames 0 and 1 are the run's largest
outliers (cosine 0.17 against the steady state, against 0.70 by frame 2, and a
gradual drift over the last three frames) — so there is no single artifact
frame at one end, and an argument from "which frame is padding" cannot settle
it. A one-block full-scale burst perturbs three to four cached frames, from
about `L` to `L+3`, with a peak that wanders (`+1, +3, +2, +3` for bursts in
blocks 12, 20, 28, 36) — the encoder attends globally, so no cached frame is a
function of local audio alone. Half a frame of ambiguity survives, 40 ms either
way, which is why the offset is a named constant and `fit` and `gate` take
`--frame-offset`: re-running E1 at 0 and comparing held-out cross-entropy is
the measurement that would close it.

The teacher's own frame accounting is *not* ambiguous and was checked, not
assumed: over 416 recorded targets, `command_encoder_frames - lead_frames`
equals `ceil(n_samples / 1280)` in **416 of 416**, both deltas exact. So the
padded cache holds the audio the teacher actually heard, and
`load_training_items` raises if a clip ever disagrees.

**A tenth of the traces were being thrown away by the harness.** One trace file
carries the whole session and the runtime's stderr is block buffered, so the
byte offset captured before a turn still has the previous turn's unflushed
frames after it. Those slices open mid-previous-turn — `t=122..121`, gap
`(122, 29)` — with this turn's frames all present behind them, and
`parse_duplex_trace` rejected the lot. It is not a speed artifact: the rate is
6-15% at 19.4 s per turn and 15.4% at 4.34 s. The pilot's 85.4% retention
simply never gated on trace parsing, so the production stage was the first
thing to see it.

The turn is the last contiguous run from the system prompt's last frame, and
`parse_duplex_trace` now takes it that way; a real hole inside the turn still
fails. Because each target stores its whole `runtime_trace`, the fix was
measured before re-running anything: re-parsing the 458 targets on disk
recovers **all 73** rejected traces, none still failing, and moves retention
from 64.9% to **78.2%** per target and 49.5% to **70.7%** paired. What remains
is barge-in (78 targets, deliberately not supervised — it stays FT_EN's own
task vector) and the reply quality gate (28). The pre-fix partial run is kept
at `targets/B1-duplex-prebugfix/`.

## Teacher generation throughput: 4.5x from not paying twice for the card

B1's original 17.7 hours were not a model limit. `-n-gpu-layers 24` was chosen
so the exec'd `voicechat-cli` could coexist with the container's own bridge
server, which holds about 14.6 GB. But the STT LM has **56 layers**, so 32 of
them (57%) were running on CPU, and the bridge is dead weight during target
generation: it serves nothing while B1 execs its own binary with an explicit
`--mmproj`.

Running the container with an idle entrypoint instead of the bridge frees the
whole card, and `-ngl 99` then fits comfortably:

| | Median turn | Per 80 ms frame |
|---|---:|---:|
| `-ngl 24`, bridge resident | 19.41 s | 225.7 ms |
| `-ngl 99`, bridge stopped | **4.34 s** | **45.7 ms** |

```bash
docker compose --env-file .cache/experiments/voice-assistant-pilot-v2/runtime.env \
  -f /tmp/nemotron-voicechat-main-229dc0e/docker-compose.yml down
docker run -d --name nemotron-voicechat --gpus all \
  -v /srv/bulk/ai/models/NemotronLabs-VoiceChat-11B-gguf:/models:ro \
  --tmpfs /tmp:size=1g -e VC_MODEL_DIR=/models -e VC_NO_BARGE=1 \
  -e VC_FORCE_BOS=1 -e VC_QUIET=10 --entrypoint sleep \
  nemotron-voicechat:f45001fc3d8013c72beb6753d3eb0b976b6a9fff infinity
```

The container keeps its name and image, so `_container_provenance` records what
it always did. Restore the bridge with `docker rm -f nemotron-voicechat` and the
usual `docker compose up -d voicechat`.

**That is the available headroom, and the rest is not worth taking.** A worker
holds 12.36 GB, so two do not fit in 24 GB, and the KV cache is not what fills
it: cutting `--session-seconds` from 180 to 40, which is still far more than the
~240 frame turns need, saved 46 MiB. GPU utilisation sits at 35-47% because the
loop is serialised per frame -- one JSON command over `docker exec` stdin, a
streaming encode of 1280 samples, a single-token decode over 56 layers, one
event back -- not because the card is saturated. Pipelining the command frames
might recover 10-15%, but writing many frames without draining stdout risks a
pipe deadlock, which is a poor trade against a run of this length.

## The teacher can answer with a tool call, and the duplex client must reply

The first full B1 duplex run stopped 17 clips from the end: alive, GPU at 0%,
no progress for 34 minutes, no error. On clip 2016 the model answered with a
tool call rather than speech --

```
frame 194: eotc, duplex tool call:
  <TOOLCALL>[{"name": "create_appointment", "arguments": {...}}]</TOOLCALL>
```

-- and the runtime blocks its serve loop until the client sends
`{"cmd":"tool_response"}` or `{"cmd":"tool_skip"}`. `DuplexPerceptionEngine`
only ever waited for `response_end`, so both sides waited; Python sat in
`anon_pipe_read`. The legacy `run_turn` path never meets this because it awaits
a single `turn_end` that already carries `tool_calls`, which is why the v1 pool
generated cleanly. It is rare -- 2 turns in 4,066 -- so 4,032 turns passed
before one appeared.

The engine now releases the loop the way `bridge/engine.py::_skip_tool_call`
does and the stage rejects that turn: like barge-in, a tool call is FT_EN's own
behaviour with no spoken reply to supervise.

**The hang should have been a timeout and was not.** `_await` computed an
absolute deadline but only re-checked it between lines, so a blocking
`readline()` ignored it entirely. `select` on the descriptor is not the fix
either: it reports the file descriptor, and would miss a line already sitting
in Python's buffered text reader. The reader now runs on a daemon thread
feeding a `queue.Queue` and `_await` waits on the queue, so the timeout is
real; a turn that does time out restarts the teacher process once rather than
costing the run.

Two things made the recovery cheap, and both were deliberate. The stage is
resumable per clip and condition, so only the missing 17 clips were
regenerated -- three minutes. And none of the fixes touch the frozen teacher
header, so `write_frozen` accepted the existing `provenance.json` unchanged and
the 4,032 targets already recorded stayed valid; changing a single string in
that header would have cost 4.8 GPU-hours. The trace log is now opened for
append, since a resumed stage had been truncating the audit history of the
turns it was resuming past.

## The encoder cache cannot share the machine with the teacher

Tried, measured, backed out. `cache-duplex` was started alongside the running
B1 teacher on the reasoning that the card had 12.2 GB free and sat at 36%
utilisation. Six minutes later it had written nothing, had not yet put the
encoder on the GPU, and had taken the teacher from **7.7 to 4.3 clips a
minute** -- a 44% loss that would push a 3 hour run past 5.

The contention is not the GPU. `arm_weights` sha256s the multi-GB VoiceChat
checkpoint and then loads it *twice* for E1, since `source` and `original` are
the same file, so the stage opens with minutes of pure disk I/O against the
same `/srv/fast` the teacher is reading its clips from, and holds two copies in
RAM (5.4 GB to 10.5 GB resident, which is also what was tripping the harness's
low-memory watchdog). Run the two stages one after the other.

Backing out cost nothing, which was the point of trying it: the stage is
resumable per clip, and it writes its `provenance.json` only after the encoder
digest it never reached, so no partial artifact was left behind.

## Open items the next session should not rediscover

- **The runtime has two turn paths and this experiment straddles them.**
  `vc_session::run_turn` is the legacy whole-wav path; `PerceptionPathEngine`
  drives it with `{"cmd":"turn","audio":...}` and it produced every B1 teacher
  trace. It honours `VC_NO_BARGE` and `VC_FORCE_BOS` and passes `a = nullptr`
  after the wav, an exact zero audio embedding.
  `vc_session::duplex_step` is what the Realtime bridge drives with
  `duplex_start` / `audio_frame`, and it is what the deployment pilot measures.
  It clears `hold_bos` and `want_bos` every frame, so neither variable applies
  and the model must emit BOS itself, and `bridge/server.py::_audio_loop` feeds
  encoded PCM silence on an input underrun rather than zeros. `runtime.env`
  flags this above the three variables: "Legacy whole-wav turn mode. The
  Realtime bridge uses the duplex stream path." Check which path a claim comes
  from before comparing it with another.
- **Teacher generation is complete.** The server was temporarily restarted for
  the v1 deployment pilot and original-control recheck, then stopped again for
  the corrected diagnostic and fitting. The original control again produced
  4/6 English calls and responded on all 12 clips. Do not leave the server
  occupying the GPU during the fit; E1 needs about 9.4 GB of 24.
- **It was previously serving Comparison 6's merge**, and had held 14.6 GB
  since that comparison finished 21 hours earlier — enough to make the fitting
  backward fail in `cublasCreate`, so the loop could only be measured after it
  was stopped. `ASR_MODEL` selects only what the container's own bridge server
  loads; B1 execs its own `voicechat-cli` with an explicit `--mmproj`, so the
  bridge is dead weight during target generation. Either encoder is one command:

  ```bash
  ASR_MODEL=container docker compose \
    --env-file .cache/experiments/voice-assistant-pilot-v2/runtime.env \
    -f /tmp/nemotron-voicechat-main-229dc0e/docker-compose.yml up -d voicechat
  ```

  `ASR_MODEL=regmean-plus-plus` restores the Comparison 6 state it was found in.
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
