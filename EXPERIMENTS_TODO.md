Use this as the implementation and experiment checklist. Complete each comparison before moving to the next.

## Shared setup

- [x] Define \(E=PT_{EN}\), \(M=PT_{ML}\), and \(F=FT_{EN}\).
- [x] Record checkpoint revisions, file hashes, configurations, and arithmetic precision.
- [x] Verify that `PT_EN` is the intended ancestor of `FT_EN`.
- [x] Restrict arithmetic to the shared canonical `encoder.*` tensors.
- [x] Verify identical tensor keys and shapes; reject broadcasting, non-finite values, or missing tensors.
- [x] Load all arithmetic/reference sources from original safetensors, perform
  arithmetic in F32, and quantize only final deployment artifacts.
- [x] Make every candidate inherit the `PT_ML` runtime configuration.
- [x] Freeze speaker-disjoint LibriSpeech map-training, validation, and test manifests.
- [x] Freeze FLEURS sentence/take manifests and require distinct English reference and query recordings.
- [x] Fit maps and select regularization using LibriSpeech only. Do not tune using final FLEURS results.
- [x] Support \(\lambda\in\{0,.25,.5,.75,1\}\), with \(\lambda=1\) designated as the primary endpoint.
- [x] Use the same evaluator and result schema for every comparison:
  - [x] English VoiceChat-space R² and cosine against `FT_EN`
  - [x] candidate-on-English retrieval
  - [x] historical centered FLEURS retrieval
  - [x] intrinsic candidate-to-candidate cross-lingual retrieval
  - [x] top-1, top-5, MRR or median rank, hit count, \(N\), and paired confidence intervals
  - [x] embedding mean/norm diagnostics
  - [x] pre- and post-quantization scores

Implemented by `shared_setup.py` and `asr_align/{experiments,manifests,evaluation}.py`.
Each experiment must materialize its own immutable `shared_setup.json` before
running; the command rejects unpinned/mismatched checkpoints or changed data.

### Speech-to-action tool calling

Every metric above is measured inside the encoder or the VoiceChat projection.
None of them shows whether the language model still understands the request and
emits the right call, and retrieval cannot show it: a foreign-language embedding
that ranks well is not evidence that the frozen LLM can read it. Every candidate
therefore also runs through the deployed server, scored at the boundary before
TTS.

The measured path is:

```text
English/Russian speech
  -> candidate perception encoder + VoiceChat projection
  -> frozen VoiceChat LLM
     |-- assistant text tokens  -> decoded transcript (scored; TTS excluded)
     `-- function tokens        -> parsed tool call -> deterministic stub result
                                                    -> post-tool assistant text
```

- [x] Score the pre-TTS boundary: decoded assistant text and the parsed
  structured call. Generated speech is recorded only as a discarded byte count.
- [x] Pair each case: the same request in English and Russian, with identical
  English tool schemas, and an expected call that is language-invariant.
- [x] Score every language against the canonical expected call, not against the
  other language's output, so a shared failure cannot look like agreement.
- [x] Make exact single-call accuracy the primary endpoint, and record tool
  attempt, well-formedness, tool name, argument types, and argument values
  separately so a failure can be located.
- [x] Record assistant text as required-fact matches rather than exact wording,
  and record English-output compliance separately from tool correctness.
- [x] Report the paired Russian-minus-English difference, Russian success
  conditional on English success, and the identical-call rate.
- [x] Serve each candidate from a pinned runtime commit and one pinned runtime
  environment, and verify through the server which encoder is loaded.
- [x] Measure the response budget from the end of the audio, so a longer clip is
  not given less decoding time than a shorter one.
- [x] Refuse to place rows scored under different manifests, prompts, budgets,
  runtimes, or precision stages in one table.
- [x] Include a control row served by the original `FT_EN` encoder, which bounds
  what the frozen LLM can do on this data and proves the harness elicits calls.
- [ ] Replace the development pilot with a frozen benchmark before any
  deployment claim: it is six single-call numeric cases, its Russian audio is
  single-speaker synthesis, and its results may be inspected before choices are
  made.

Implemented by `voice_assistant_evaluation.py` and `asr_align/voice_assistant.py`.
Retrieval remains screening evidence; this is the endpoint the alignment exists
to move.

The four rows through Comparison 3 are recorded in the ignored pilot output
named in each `run.json`, with the combined table in
`voice-assistant-pilot-v2/analysis-through-comparison-3/`. The `FT_EN` control calls
the correct tool on four of six English cases and answers all six Russian clips
in English without ever attempting a call. Comparisons 1 and 2 produce no
assistant turn at all in either language, so their per-language and paired scores
are zero for a different reason than the control's Russian zeros, which the
`Responded` column separates. Comparison 3 recovers one exact English call and
its correct English answer; the other eleven clips produce no assistant turn.

## 1. `PT_ML` baseline

- [x] Load the unmodified `PT_ML` encoder.
- [x] Attach the original VoiceChat/`FT_EN` projection without an alignment map.
- [x] Run a deterministic forward-pass sanity check.
- [x] Export and reload a pass-through copy.
- [x] Verify that the pass-through artifact matches the original model before quantization.
- [x] Measure any change caused by export and deployment quantization.
- [x] Reproduce the historical French, German, and Russian FLEURS results using the frozen manifest.
- [x] Run the complete shared evaluation.
- [x] Save pooled embeddings and per-layer activations for reuse.
- [x] Record this result as the reference against which comparisons 2–5 are measured.
- [x] Run the paired speech-to-action tool-calling evaluation on the deployment artifact.

Done when the baseline is reproducible and its artifact, manifest, metrics, and hashes are recorded.

Implemented by `pt_ml_baseline.py` and `asr_align/baseline.py`. The boxes above
remain unchecked until the runner completes against a real frozen setup and
the recorded pre/post artifacts and metrics validate.

## 2. Direct task arithmetic

Construct:

\[
\Delta_F=F-E,\qquad
C_\lambda=M+\lambda\Delta_F.
\]

- [x] Compute the encoder-only task vector `FT_EN − PT_EN`.
- [x] Record task-vector norms by block, module type, and tensor.
- [x] Verify the reconstruction invariant:
  \[
  E+(F-E)\approx F.
  \]
- [x] Verify that \(\lambda=0\) exactly reproduces the `PT_ML` baseline.
- [x] Construct candidates for every predefined \(\lambda\).
- [x] Keep the original VoiceChat projection unchanged.
- [x] Run forward checks for finite outputs and abnormal activation/norm growth.
- [x] Run the complete shared evaluation for every \(\lambda\).
- [x] Quantize and reevaluate the primary \(\lambda=1\) artifact.
- [x] Plot or tabulate the English-transfer versus multilingual-retention Pareto curve.
- [x] Do not select a final \(\lambda\) yet; preserve all development results.
- [x] Run the paired speech-to-action tool-calling evaluation on the primary
  \(\lambda=1\) deployment artifact.

Done when direct arithmetic has a validated \(\lambda\)-sweep and a deployment-precision result at \(\lambda=1\).

Implemented by `direct_task_arithmetic.py` and `asr_align/direct.py`. The
validated run is recorded under the ignored experiment output named in its
`run.json`; every result is paired against the exact frozen Comparison 1 arrays.

## 3. Final activation-map projection only

Keep the `PT_ML` encoder unchanged and learn only the final activation correspondence.

- [x] Collect paired final-layer activations from `PT_EN` and `PT_ML` on LibriSpeech map-training data.
- [x] Center activations using training-set statistics.
- [x] Fit bidirectional ridge maps:
  \[
  h_EA_L\approx h_M,\qquad h_MB_L\approx h_E.
  \]
- [x] Regularize the maps toward identity.
- [x] Select regularization using held-out LibriSpeech speakers.
- [x] Record held-out R², cosine, condition number, singular values, distance from identity, and cycle consistency.
- [x] Test whether both maps generalize to foreign FLEURS activations without refitting.
- [x] Fold the reverse map into the VoiceChat projection:
  \[
  W_{\text{proj},M}=W_{\text{proj},F}B_L^\top.
  \]
- [x] Compose any final affine offset into the projection bias.
- [x] Confirm that encoder tensors remain byte-identical to `PT_ML`.
- [x] Run the complete shared evaluation.
- [x] Compare directly with comparison 1 to isolate the effect of interface alignment.
- [x] Quantize and reevaluate the mapped projection.
- [x] Run the paired speech-to-action tool-calling evaluation on the deployment
  artifact. Its exported directory must carry the folded `proj.*` and featurizer
  tensors, or the deployment converter silently falls back to the container's
  own projection and invalidates the result.

Done when the benefit and cross-lingual cost of final-layer alignment are measured independently of task-vector fusion.

Implemented by `final_map_projection.py` and `asr_align/final_map.py`. The
validated run is recorded under the ignored experiment output named in its
`run.json`; `PT_ML`'s side of the paired activations is the frozen Comparison 1
cache, which reproduced bit-for-bit when one shard per split was re-encoded, and
every result is paired against the exact frozen Comparison 1 arrays.

Both maps are identity-regularized ridge fits over 57,912 `map_train` frames,
with the penalty chosen on 21,660 held-out `validation` frames by target-space
R² rather than by the VoiceChat-space R² the shared evaluator reports, because
the frozen validation split is also the evaluation split. Selected
alpha = 0.001 forward and 0.1 reverse; held-out R² is +0.44 in both directions
against -25.6 and -1.1 for the untouched interface.

What the arm buys, pre- and post-quantization alike, is the English interface:
VoiceChat-space R² against `FT_EN` moves from -0.701 to -0.024 (paired 95% CI
[+0.6705, +0.6827]) and cosine from 0.301 to 0.541 ([+0.2385, +0.2413]).
Intrinsic multilingual retrieval is unchanged to the digit, which is not luck:
it is measured before the projection and the encoder is byte-identical to
`PT_ML` (SHA-256 over all 636 canonical tensors matches the source, the export,
and the Comparison 1 artifact). Historical centered FLEURS top-1 and MRR
differences have paired intervals containing zero. Post-quantization French
top-5 improves by 0.0677, CI [0.0075, 0.1278]; the other historical top-5
intervals include zero. Interface alignment therefore improves this English
embedding metric more than any Comparison 2 coefficient while preserving the
measured intrinsic retrieval. It does not establish downstream understanding.

The FLEURS generalization test is the honest limit. Both maps hold on English
FLEURS takes (forward R² +0.36, reverse +0.29) but fall on foreign speech, where
the reverse map goes negative (-0.31 de, -0.53 fr, -0.67 ru) -- worse than
predicting the mean, though still well above the untouched interface's -1.8 to
-2.3. The map was fitted on English speech only. Foreign scores compare PT_EN
and PT_ML activations on the same recording; they do not measure foreign
transcription accuracy.

The deployed Q8 candidate makes 1/6 exact English calls and 0/6 Russian calls
on the frozen development pilot. Its single response calls `math.factorial`
with `number=5` and correctly answers 120 in English; all other clips exhaust
the response budget without a turn, with no transport errors. The paired
Russian-minus-English difference is -0.1667, 95% CI [-0.5, 0.0]. The FT_EN
control remains at 4/6 English calls. Large embedding gains thus restore only
one pilot response and do not support a deployment claim.

[The Comparison 3 report](COMPARISON_3_RESULTS.md) indexes the frozen artifacts,
commands, hashes, metrics, and completion evidence. Both precision-stage
results and their paired intervals reproduced exactly from saved embeddings.
Runtime parity passed on 85 frames with a maximum sampled discrepancy of
0.0527 embedding standard deviations, within the checker's documented 0.10
tolerance for fitted projections. All 60 unit tests passed. The earlier
fitting run remains immutable; completion evidence is saved separately in
`.cache/experiments/comparison-3-final-map-v1-completion/`.

## 4. Dense activation-transported task vector

> `LITERATURE.md` §3.3 proposes demoting comparisons 4 and 5 in favour of
> comparison 6, on the grounds that closed-form merging reaches the same
> output-matching objective without the structured-tensor routing problem. That
> is a planning proposal only. Neither comparison is superseded until a decision
> is recorded in `AGENTS.md`.

Learn activation maps for every relevant internal representation and use them to transport the fine-tuning delta.

- [ ] Collect paired `PT_EN`/`PT_ML` activations for:
  - residual boundaries
  - both FFN hidden spaces in every block
  - convolution-channel spaces
  - attention heads and any required head-internal spaces
- [ ] Fit bidirectional, identity-regularized maps \(A_g:E\rightarrow M\) and \(B_g:M\rightarrow E\) for every group.
- [ ] Select regularization on held-out LibriSpeech speakers.
- [ ] Record map quality, conditioning, identity distance, and cycle consistency per group.
- [ ] Evaluate every map on foreign FLEURS activations before using it for fusion.
- [ ] Mark groups whose maps are unstable, ill-conditioned, or English-specific.
- [ ] For each dense linear or pointwise-convolution task delta, compute:
  \[
  \Delta W_M=A_{\text{out}}^\top\Delta W_EB_{\text{in}}^\top.
  \]
- [ ] Transport bias deltas with:
  \[
  \Delta b_M=A_{\text{out}}^\top\Delta b_E.
  \]
- [ ] For LayerNorm, depthwise convolution, GLU partitions, and attention structure, solve for the closest architecture-preserving parameter update on LibriSpeech rather than creating unsupported dense operations.
- [ ] Record the approximation residual for every structured update.
- [ ] Construct:
  \[
  C_\lambda^{AT}=M+\lambda\Delta_F^{AT}.
  \]
- [ ] Use the final reverse activation map in the VoiceChat projection.
- [ ] Verify \(\lambda=0\) against comparison 3.
- [ ] On held-out data, test the defining condition:
  \[
  h_l^{C_\lambda^{AT}}-h_l^M
  \approx
  \lambda(h_l^F-h_l^E)A_l.
  \]
- [ ] Report this transported-update agreement at every layer.
- [ ] Run the complete shared evaluation for every \(\lambda\).
- [ ] Quantize and reevaluate the primary \(\lambda=1\) artifact.
- [ ] Run the paired speech-to-action tool-calling evaluation on the primary
  \(\lambda=1\) deployment artifact, exported with its own projection.

Done when both the endpoint metrics and the layerwise transported-update approximation have been measured. Do not treat the method as successful solely because the base activation maps have high R².

## 5. Hybrid activation transport with structured deltas applied directly

Reuse comparison 4’s activation maps, but apply structurally incompatible parameter deltas indexwise.

- [ ] Start from the exact maps and hyperparameters selected in comparison 4.
- [ ] Transport dense linear, pointwise-convolution, and compatible bias deltas using \(A_{\text{out}}\) and \(B_{\text{in}}\).
- [ ] Apply the following `FT_EN − PT_EN` deltas directly by index:
  - LayerNorm scale and bias
  - depthwise-convolution kernels
  - any attention parameters that cannot preserve their head structure under the dense maps
  - other explicitly classified structured tensors
- [ ] Produce a manifest classifying every encoder tensor as:
  - activation-transported
  - directly added
  - omitted
- [ ] Assert that every encoder tensor appears exactly once in the manifest.
- [ ] Construct every predefined \(\lambda\) candidate.
- [ ] Use the same final reverse-map projection as comparisons 3 and 4.
- [ ] Verify \(\lambda=0\) against comparison 3.
- [ ] Measure layerwise transported-update agreement using the same test as comparison 4.
- [ ] Run the complete shared evaluation for every \(\lambda\).
- [ ] Quantize and reevaluate the primary \(\lambda=1\) artifact.
- [ ] Compare comparison 5 against:
  - comparison 2 to measure the value of dense activation transport;
  - comparison 4 to measure whether direct structured deltas outperform architecture-projected structured deltas;
  - comparison 3 to verify that improvement is not explained only by the final activation map.
- [ ] Run the paired speech-to-action tool-calling evaluation on the primary
  \(\lambda=1\) deployment artifact, exported with its own projection.

Done when the hybrid/full-transport difference is isolated and all tensor-routing decisions are reproducible.

## 6. RegMean++ merge with the original projection

Merge `PT_ML` and `FT_EN` in closed form and evaluate through the untouched
VoiceChat interface. Training-free. This is arm **E0**, and it is the only new
arm that needs no invariant change: `proj` is preserved exactly as invariant 6
requires. Design record and rejected alternatives: `REGMEAN_INTERFACE_DESIGN.md`.

- [ ] Freeze a SLURP manifest (English assistant audio) and a Speech-MASSIVE
  fr/de/ru manifest through `manifests.write_frozen`, with the same
  content-addressing and per-file SHA-256 verification as the existing
  manifests. FLEURS must not appear in either.
- [ ] Size Dataset A by dimension, not by the paper's literal 256 samples:
  at least 4× the largest linear input dimension in frames, i.e. roughly
  20--30 minutes of audio per candidate. Confirm `n_ff` = `intermediate_size`
  from the configuration before fixing the count.
- [ ] Collect \(G_F\) on SLURP `train` and \(G_M\) on Speech-MASSIVE `dev`,
  kept separate. Never collect both on a shared English domain: with equal Gram
  matrices Eq. 2 reduces exactly to the unweighted mean.
- [ ] Normalize each \(G_i\) by frame count and equalize total frames across
  candidates, so neither clip length nor collection volume acts as an
  unintended merge coefficient.
- [ ] Classify every encoder tensor as RegMean-merged or averaged, and assert
  each appears exactly once. `conv.pointwise_conv{1,2}` are 1×1 convolutions
  and belong on the RegMean path; `depthwise_conv`, the subsampling
  convolutions, LayerNorms, `bias_u` and `bias_v` are averaged.
- [ ] Implement RegMean++ Algorithm 1: for each depth, obtain the cross-layer
  input from the **merged** prefix, run each candidate's own layer on it for the
  intra-layer sub-module inputs, then solve Eq. 2 per linear layer. Budget \(L\)
  sequential forward passes; the cached comparison 1 activation shards are
  candidate-only and are **not** sufficient for the ++ variant.
- [ ] Grid \(\alpha\in\{0.1,0.3,0.5,0.7,0.9,0.95\}\) on a held-out split.
  Exclude \(\alpha=1.0\).
- [ ] Record the merged depth range and module subset as declared choices. The
  paper reports that middle and deep layers preserve >98% of the all-layer
  result and that MLP linears outperform attention linears.
- [ ] Build plain RegMean as a reference and simple averaging as arm **E4**'s
  encoder, to separate the cross-layer correction from the Gram weighting.
- [ ] Ablate LayerNorm handling: averaged, versus seeded from `FT_EN`.
- [ ] Ablate the Gram data: re-collect \(G_M\) from CoVoST 2 / Common Voice
  fr-de-ru and re-merge, replicating the paper's in-domain versus out-of-domain
  sensitivity on this model.
- [ ] Confirm the merged candidate inherits `PT_ML`'s complete runtime
  configuration, including the 56-frame left context. Record that `FT_EN`'s
  Gram contribution is therefore collected at a context it never saw.
- [ ] Run the complete shared evaluation.
- [ ] Quantize and reevaluate.
- [ ] Run the paired speech-to-action tool-calling evaluation on the deployment
  artifact.
- [ ] Compare against comparison 1 and comparison 2 to isolate closed-form
  merging from task arithmetic.

Done when a training-free merge has been measured end to end through the
original interface. Do not treat per-layer regression residuals as evidence of
success: RegMean++ solves each layer greedily and does not control error at the
encoder output, which is the only quantity the frozen language model reads.

## 7. End-to-end interface fitting, and the merge ablation

Fit `proj` by token cross-entropy through the frozen VoiceChat language model,
and use it to ablate whether the merge of comparison 6 was needed at all.

**Blocked** until a decision extending invariant 6 is recorded in `AGENTS.md`:
all four arms train `proj` by gradient descent, which goes beyond the current
"learned reverse activation map" clause. Extend once for all arms, not per arm.

**Blocked** until the gating check passes: the frozen language model must read
fr/de/ru text and answer in-language when instructed. If it will not, condition
B has no teacher and must be reported as unavailable rather than trained.

- [ ] Run the gating check: frozen LM, text-only, ~100 MASSIVE utterances per
  language, both system prompts. Record whether it answers sensibly from native
  text, replies in the input language under prompt B, and replies in English
  under prompt A.
- [ ] Freeze two system prompts as separate conditions: **A** "reply in English
  only" and **B** "reply in the input language". Under invariant 9 these are two
  comparison rows, not one row with a prompt column.
- [ ] Build Dataset B on clips disjoint from Dataset A:
  - **B1**, SLURP `val` audio, target = what the original VoiceChat emits on the
    same clip. The original `proj` reaches zero loss on B1 by construction.
  - **B2**, Speech-MASSIVE `dev` remainder, target = the frozen LM run text-only
    on the **native-language** transcript under the condition's prompt.
- [ ] Generate B2 targets from the target-language MASSIVE text, never from
  `en-US`. MASSIVE localized rather than translated, so an `en-US`-derived
  target names entities the foreign audio never contained.
- [ ] Record MASSIVE's per-slot replacement method per utterance; it identifies
  where the two output conditions are most likely to diverge.
- [ ] Measure the text-path versus audio-path target gap on a small English
  subset, so B1 and B2 losses are on a comparable scale.
- [ ] Gate targets on output-language identification and report teacher quality
  separately, establishing the distillation ceiling.
- [ ] Train one shared `proj` across both conditions. `proj` never sees the
  system prompt, so a single projection serves both; fitting condition A alone
  would reward discarding language identity, which condition B forbids.
- [ ] Keep English clips under both prompts as the control separating prompt
  effect from input-language effect.
- [ ] Precompute the frozen encoder's output once per clip and cache it, and
  cache one prefix KV per system prompt. Neither depends on `proj`.
- [ ] Run the four arms:
  - **E1** `FT_EN` + trained `proj`, initialized at the original projection.
  - **E2** `PT_ML` byte-identical + trained `proj`, initialized at comparison
    3's folded map.
  - **E3** the comparison 6 merge + trained `proj`, initialized at the original
    projection.
  - **E4** simple averaging + trained `proj`, initialized at the original
    projection.
- [ ] Run E1 first. It is a required control, not an optional extra: B2 targets
  come from a multilingual language model, so foreign-audio performance must be
  bounded against an English-only encoder before any of it is attributed to
  `PT_ML`. It is also the pipeline check — if E1 does not at least match the
  `FT_EN` control row on English, the training loop is broken and nothing
  downstream is meaningful.
- [ ] Run each arm at 25/50/100% of Dataset B and compare scaling curves, not
  endpoints. The arms are asymmetrically data-sensitive: E3 needs a small
  correction from a good prior while E2 must learn a much larger re-basing, so
  at a fixed budget E3 is flattered. Dataset B is ~6.7 hours of unique audio
  against SLAM-ASR's ~960.
- [ ] Record the language-model precision used for fitting in each candidate's
  provenance, and measure the NF4-versus-bf16 fitting gap on a few hundred
  examples. Deployment runs Q8_0, so the fitting precision is an experimental
  variable, not only an export stage.
- [ ] Confirm encoder tensors remain byte-identical to their arm's source.
- [ ] Run the complete shared evaluation for every arm and condition.
- [ ] Score the held-out Speech-MASSIVE `test` split on both axes: output
  language identification and intent/slot correctness from the inherited MASSIVE
  labels.
- [ ] Quantize and reevaluate.
- [ ] Run the paired speech-to-action tool-calling evaluation per condition.

Done when the four arms have been measured under both conditions with matched
data budgets. E2 winning is a legitimate outcome and would mean the interface,
not the merge, was the binding constraint, as `LITERATURE.md` §2.4 predicts. Do
not report a merge advantage from endpoint scores alone.

## Final comparison

- [ ] Produce one table containing comparisons 1–7 under identical manifests and precision.
- [ ] Report paired differences relative to `PT_ML`, not only absolute scores.
- [ ] Separate results into:
  - English fine-tune transfer
  - intrinsic multilingual retention
  - VoiceChat-space cross-lingual alignment
  - layerwise transport fidelity
  - quantization sensitivity, and separately the language-model precision used
    for interface fitting
  - interface fitting under both output-language conditions, reported as two
    rows per invariant 9
  - Speech-MASSIVE output-language identification and intent/slot correctness
  - Dataset B scaling curves at 25/50/100%, not endpoint scores alone
  - paired English/Russian speech-to-action tool calling, against the `FT_EN`
    control row
- [ ] Select a candidate only from development results.
- [ ] Evaluate the selected candidate once on the reserved final split.
- [ ] Treat retrieval as screening evidence and run actual ASR/VoiceChat evaluation before making a deployment claim.
