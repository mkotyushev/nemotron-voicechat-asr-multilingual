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

The first three rows are recorded in the ignored pilot output named in each
`run.json`, with the combined table in its `analysis/`. The `FT_EN` control calls
the correct tool on four of six English cases and answers all six Russian clips
in English without ever attempting a call. Comparisons 1 and 2 produce no
assistant turn at all in either language, so their per-language and paired scores
are zero for a different reason than the control's Russian zeros, which the
`Responded` column separates.

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

- [ ] Collect paired final-layer activations from `PT_EN` and `PT_ML` on LibriSpeech map-training data.
- [ ] Center activations using training-set statistics.
- [ ] Fit bidirectional ridge maps:
  \[
  h_EA_L\approx h_M,\qquad h_MB_L\approx h_E.
  \]
- [ ] Regularize the maps toward identity.
- [ ] Select regularization using held-out LibriSpeech speakers.
- [ ] Record held-out R², cosine, condition number, singular values, distance from identity, and cycle consistency.
- [ ] Test whether both maps generalize to foreign FLEURS activations without refitting.
- [ ] Fold the reverse map into the VoiceChat projection:
  \[
  W_{\text{proj},M}=W_{\text{proj},F}B_L^\top.
  \]
- [ ] Compose any final affine offset into the projection bias.
- [ ] Confirm that encoder tensors remain byte-identical to `PT_ML`.
- [ ] Run the complete shared evaluation.
- [ ] Compare directly with comparison 1 to isolate the effect of interface alignment.
- [ ] Quantize and reevaluate the mapped projection.
- [ ] Run the paired speech-to-action tool-calling evaluation on the deployment
  artifact. Its exported directory must carry the folded `proj.*` and featurizer
  tensors, or the deployment converter silently falls back to the container's
  own projection and invalidates the result.

Done when the benefit and cross-lingual cost of final-layer alignment are measured independently of task-vector fusion.

## 4. Dense activation-transported task vector

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

## Final comparison

- [ ] Produce one table containing comparisons 1–5 under identical manifests and precision.
- [ ] Report paired differences relative to `PT_ML`, not only absolute scores.
- [ ] Separate results into:
  - English fine-tune transfer
  - intrinsic multilingual retention
  - VoiceChat-space cross-lingual alignment
  - layerwise transport fidelity
  - quantization sensitivity
  - paired English/Russian speech-to-action tool calling, against the `FT_EN`
    control row
- [ ] Select a candidate only from development results.
- [ ] Evaluate the selected candidate once on the reserved final split.
- [ ] Treat retrieval as screening evidence and run actual ASR/VoiceChat evaluation before making a deployment claim.
