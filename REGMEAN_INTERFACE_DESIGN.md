# Design record: RegMean++ merging and end-to-end interface fitting

Status: **design, plus the §11 gating checks.** No merge, fit or candidate in
this document has been run. The blocking checks of §11 have: their results are
in `COMPARISON_7_GATE_RESULTS.md` and are summarised at the head of that
section. Everything else specifies comparisons 6 and 7 of `EXPERIMENTS_TODO.md`
and records the reasoning and the rejected alternatives behind them, so that the
specification can be audited without reconstructing the argument.

Scope. Comparisons 1--3 established that the *interface* — the 1024→4480
`proj` into the VoiceChat language model's embedding space — is the currently
binding constraint, not the weight-space merge (`LITERATURE.md` §2.4). Both
comparisons here follow from that: comparison 6 is a training-free merge
evaluated through the untouched interface, comparison 7 refits the interface
end to end and uses that to ablate whether the merge was needed at all.

Notation is `README.md`'s: `E = PT_EN`, `M = PT_ML`, `F = FT_EN`.

---

## 1. The method

RegMean (Jin et al., ICLR 2023) merges each linear layer by least squares
against the candidates' own outputs. For linear layer \(j\) in transformer layer
\(l\), with \(X_i\) the input features of candidate \(i\):

\[
\mathcal{L} = \sum_i \lVert X_i W_M - X_i W_i\rVert^2
            + \sum_i \operatorname{tr}\!\left[(W_M-W_i)^\top \Lambda_i (W_M-W_i)\right],
\qquad
\Lambda_i = \tfrac{1-\alpha}{\alpha}\operatorname{diag}(X_i^\top X_i)
\]

with the closed form

\[
W_M = \Big[\sum_i \hat G_i\Big]^{-1}\sum_i \hat G_i W_i,
\qquad
\hat G_i = \alpha\,X_i^\top X_i + (1-\alpha)\operatorname{diag}(X_i^\top X_i).
\tag{Eq. 2}
\]

RegMean++ (Nguyen et al., TMLR 2026, `papers/2508.03121v3.pdf`) changes only how
\(X_i\) is obtained. RegMean takes it from candidate \(i\)'s own forward pass;
RegMean++ takes the cross-layer input from the **merged** model's previous layer
(Algorithm 1, line 3: \(X_i^{(l)} \leftarrow f_M^{(l-1)}(X_i^{(l-1)})\)), then
runs candidate \(i\)'s own layer \(l\) on it (line 4) to get the intra-layer
sub-module inputs. Non-linear parameters are merged by simple averaging.

There is **no gradient descent anywhere in the merge**. Per linear layer it is
one symmetric solve. The only passes are forward passes.

---

## 2. Decision: Gram matrices come from per-candidate domain data

**Decided.** `G_M` is collected on multilingual audio, `G_F` on English
assistant audio, kept separate and summed only inside Eq. 2.

**Rejected: collecting both on the shared `PT_EN` English domain.** This was the
initial instinct and it is a degenerate configuration. Eq. 2 is a
\(\hat G\)-weighted average of the candidate weights; if the Gram matrices
coincide it reduces exactly to the unweighted mean:

\[
\hat G_1 = \dots = \hat G_K = \hat G
\;\Longrightarrow\;
W_M = [K\hat G]^{-1}\hat G\sum_i W_i = \tfrac{1}{K}\sum_i W_i,
\]

independent of \(\hat G\). Every bit of information RegMean adds over Model Soup
lives in the *difference* between the candidates' input second-moment matrices,
and common data is precisely what erases it.

This binds harder on ++ than on plain RegMean. Under RegMean the Grams still
differ on shared audio because the candidate weights differ. Under RegMean++ the
cross-layer input is the merged model's, so on identical audio it is identical
across candidates; the only residual asymmetry is each candidate's own
intra-layer sub-modules (for Q/K/V, just its LayerNorm). **The ++ correction
removes the very signal source the common-data plan relied on.**

Empirically the paper's §5.8 / Table 5 shows regression-based merging is
sensitive in the same direction: substituting off-domain ImageNet samples drops
RegMean++ on ViT-B/32 from 84.4 to 65.5, while Fisher merging is stable. Their
wording: *"a limitation of regression-based methods when the merging data
distribution is misaligned with the task domains."*

**Why the distillation intuition pointed the wrong way.** It is a correct
intuition attached to the wrong method. Fitting a *paired map* (comparison 3:
ridge from `PT_ML` space to `FT_EN` space) does require common audio, because
pairs must come from the same utterance. RegMean forms no pairs — it reweights
each candidate's own weights, and the data only says where in input space that
candidate is the authority. The rule: **paired-map fitting wants common data;
covariance-weighted merging wants per-domain data.**

Note also that feeding `F` foreign audio would be actively harmful, not merely
uninformative: the recorded pilot has `F` mishearing all six Russian clips, and
Eq. 2 would be asked to preserve that behaviour.

**Open knob, not yet specified.** Scaling \(G_i\) by a per-candidate constant
biases the solution toward one candidate. It is the natural analogue of the
\(\lambda\) sweep and would need an explicit extension of invariant 7 before use.

---

## 3. Decision: no initialisation problem exists

**Decided.** The merged model is built layer by layer in depth order:
non-linear tensors are simple-averaged, linear tensors are overwritten by Eq. 2,
each depth consuming activations from the already-merged prefix. Random
initialisation is not applicable and would be meaningless.

`PT_EN` plays **no role**. RegMean merges full weights \(W_i\), not deltas, so
unlike task arithmetic it needs no shared base. Consequently `LITERATURE.md`
§2.2's concern — heavy continued pre-training degrading linear mode
connectivity — does not apply in the same form: the method assumes positional
correspondence between layers, not a shared basin.

### 3.1 What counts as a linear layer in this encoder

From `asr_align/encoder.py`:

| Tensor | Treatment | Note |
|---|---|---|
| `self_attn.{q,k,v,o}_proj`, `relative_k_proj` | RegMean | `n_embd`→`n_embd` |
| `feed_forward{1,2}.linear{1,2}` | RegMean | `linear2` input is `n_ff` = `intermediate_size` |
| `conv.pointwise_conv{1,2}` | **RegMean** | 1×1 `Conv1d` *is* a dense linear over channels; reshape to `(out, in)`. Easy to leave on the averaging path by mistake. |
| `subsampling.linear` | RegMean-able | `d_in` = 4352; see §3.2 |
| `subsampling.conv*`, `conv.depthwise_conv` | average | grouped/spatial, no dense form |
| all `LayerNorm`, `bias_u`, `bias_v` | average | |

Note `nn.Linear` stores `(out, in)`; the \(W\) of Eq. 2 is its transpose, and
\(G\) is over the input dimension.

### 3.2 Depth and module selectivity

The paper's contribution ② reports that merging linear layers from **only the
middle and deep transformer layers preserves >98%** of the all-layer result,
that early-layer linears degrade it, and that **MLP linears consistently
outperform attention linears**. This coincides with the repository's own
measurement that `F`/`E` and `E`/`M` weight cosines diverge with depth. The
specification therefore treats the depth range and the module subset as declared
choices to be recorded per candidate, not as defaults.

### 3.3 Non-linear tensors matter more under ++ than under RegMean

Because RegMean++ feeds merged activations forward, the averaged non-linear
tensors influence every downstream solve. REPAIR and *Vanishing Feature* both
identify normalisation as decisive for whether a merged network recovers.
Seeding the backbone's LayerNorms from `F` instead of averaging is therefore a
cheap and defensible ablation, and is specified as one.

---

## 4. Decision: no ground-truth loss inside the merge

**Decided.** No output-level or label loss enters comparison 6. RegMean is
training-free and label-free; adding a GT term destroys the closed form and
converts the merge into gradient-based multi-task training.

**But the stated reason for omitting it was wrong and the correction matters.**
RegMean++ does *not* give end-to-end output matching. Each layer is still solved
greedily: it conditions on the merged prefix but never looks downstream and
never sees the network output. The objective remains \(\sum_l\)(local
regression), so residual error at the encoder output — the only quantity the
frozen language model actually reads — is uncontrolled. The paper's own Figures
3--4 show CKA similarity still decaying with depth under ++, better than
RegMean but not flat.

This is exactly why comparison 7 exists as a separate slot rather than as an
option inside comparison 6.

Note also that `F` has no ground truth in the ordinary sense: its output is an
embedding consumed by the language model, not a class or token distribution. A
GT loss requires the whole VoiceChat language model in the loop.

---

## 5. Decision: Gram sample size is derived from dimension, not copied

**Decided.** Target **≥4× the largest input dimension in frames**, i.e. roughly
20--30 minutes of audio per candidate (~400--700 short assistant clips).

The paper's default of "256 samples" is 256 *images* × ~50 patches ≈ 12.8k rows
against a largest \(d_{in}\) of 3072 — a ratio of about 4. Copying the literal
256 to speech is wrong: at 12.5 frames/sec, 256 three-second clips give ~9.6k
frames, so `feed_forward*.linear2` (\(d_{in}\) = `n_ff`, 4096 if it is the usual
4×`n_embd` — **confirm from the config**) would be solved at a ratio of 2.3 and
`subsampling.linear` (\(d_{in}\) = 4352) at 2.2. Those Grams are thin and the
\(\alpha\) shrinkage would silently carry far more weight than intended.

Two further requirements the paper does not cover, because its samples are
fixed-size and speech is not:

- **Normalise each \(G_i\) by frame count.** A 30 s clip otherwise contributes
  4× a 7 s clip, making the clip-length distribution an unintended merge
  coefficient.
- **Equalise total frames across candidates**, or the relative candidate
  weighting is an artifact of how much audio happened to be collected.

\(\alpha\) is grid-searched over \(\{0.1,0.3,0.5,0.7,0.9,0.95\}\) on a held-out
split. The paper found 0.95 optimal for ViTs but only 0.1--0.3 for
Llama/Gemma, so neither end is assumed. \(\alpha = 1.0\) (no shrinkage) zeroes
out accuracy in their Table 9 and is excluded.

---

## 6. Decision: dataset composition

Three corpora chain together because **MASSIVE is a professional localisation of
SLURP**, and **Speech-MASSIVE is the recorded speech counterpart of MASSIVE**,
linked 1-to-1 by utterance id.

### Dataset A — Gram collection (unlabelled, no targets)

| | Source | Draw |
|---|---|---|
| \(G_F\) | SLURP `train` | English assistant commands |
| \(G_M\) | Speech-MASSIVE `dev`, fr/de/ru | same task, other languages |

Because both sides are the same 18 domains / 60 intents in different languages,
the Gram difference isolates **language and acoustics** rather than task. Under
Eq. 2 that means the merge is weighted along the intended axis and is not
confounded by domain shift.

**Specified ablation.** Re-collect \(G_M\) from CoVoST 2 / Common Voice fr-de-ru
(general read speech, off-domain for the assistant task) and re-merge. This
replicates the paper's Table 5 ID-vs-OOD sensitivity on this model for the cost
of one extra merge.

### Dataset B — interface fitting (text targets through the frozen LM)

Two terms, on **unaligned clips** — no cross-language audio pairing is needed
anywhere, which removes the hardest data constraint.

- **B1, interface anchor (English).** SLURP audio held out from A. Target is
  what the *original* VoiceChat (`F` encoder + original `proj`) emits on that
  same clip. Pure self-distillation, no annotation. Key property: **the original
  `proj` achieves zero loss on B1 by construction**, so initialising there
  starts this term at 0 and it acts as an interface regulariser with a
  meaningful floor.
- **B2, multilingual transfer.** Speech-MASSIVE fr/de/ru `dev` minus the A draw.
  Target is the frozen LM run text-only on the transcript, under the condition's
  system prompt.

---

## 7. Decision: prompt the teacher with the native transcript

**Decided.** B2 targets are generated by feeding the frozen LM the **target
language** MASSIVE text (`ru-RU`, `fr-FR`, `de-DE`), never the `en-US` text.

**This supersedes an earlier plan** to route targets through the `en-US` text
plus a localisation filter. That plan had a real defect. MASSIVE did not
translate, it *localised*: entities were deliberately swapped per language, so an
English utterance naming *John* becomes *Иван* in Russian. Generating the target
from the `en-US` side and training the Russian audio path toward it teaches the
projection to emit entities the audio never contained — a hallucination
objective that would train cleanly and fail silently.

Prompting from the native transcript removes the mismatch for both output
conditions at once, needs no filter, and — because Speech-MASSIVE audio is a
recording of exactly that text — also removes ASR error from the teacher path.
MASSIVE's `en-US` side remains useful only as a scoring reference.

MASSIVE's per-slot replacement-method annotations (`translation` /
`localization` / `unchanged`) are no longer needed as a filter. They are still
worth recording per utterance, as they identify the subset where the two output
conditions are most likely to diverge.

---

## 8. Decision: both output-language conditions, one shared projection

**Decided.** Two frozen system prompts — **A: "reply in English only"**,
**B: "reply in the input language"** — and a single `proj` trained across both.

**The argument is not primarily about measurement coverage.** Under condition A
alone, *discarding language identity is a loss-reducing solution*: the target
never depends on the input language, so a projection collapsing fr/de/ru frames
onto their nearest English-sounding embedding would score better. That is a
degenerate shortcut which trains cleanly and defeats the purpose of the
multilingual encoder. Condition B's targets can only be produced if language
identity survives into the embedding space, so training on both makes
preservation a requirement rather than an option. Condition B is a regulariser
first and a metric second.

Forcing English output on foreign input also makes the model translate
internally, which can degrade content fidelity independently of the merge.

**One projection, not two.** `proj` never sees the system prompt — it maps
encoder frames to embedding space and nothing else. The prompt varies only the
target, hence the gradient. Two separate fits would give less diverse
supervision on the same 4.59M parameters and would let each specialise to its
output-language convention. Cost is two cached prefix KVs instead of one.

English clips are kept under **both** prompts even though their targets should
coincide; that pair is the control separating "the prompt changed behaviour"
from "the input language changed behaviour".

**Blocking prerequisite — discharged.** This design assumed the frozen LM reads
fr/de/ru *text* competently and will answer in-language when instructed. The
recorded pilot only showed it answering in English from misheard audio, which
said nothing about text-mode multilingual ability. §11 has now measured it: it
does, through the chat format, on 0.90/0.85/0.82 of fr/de/ru utterances — and it
does not through the deployment runtime's perception-channel text path, where
Russian collapses to 0.07. Condition B exists; its teacher is the chat format.

---

## 9. Decision: the arms, and what each one isolates

| Arm | Encoder | `proj` | Isolates |
|---|---|---|---|
| **E0** | RegMean++ merge | original, frozen | the merge alone, training-free — comparison 6 |
| **E1** | `FT_EN` | trained | pipeline sanity; the multilingual floor owed to LM priors alone |
| **E2** | `PT_ML`, byte-identical | trained, init = comparison 3's folded map | whether the merge is needed at all |
| **E3** | RegMean++ merge | trained, init = original | whether merging beats retraining the interface |
| **E4** | simple average | trained, init = original | whether the Gram weighting earned its complexity |

**E0** is the only new arm that needs no governance change: it leaves `proj`
untouched and so stays inside invariant 6 as written. It runs first.

**E1 is a required control, not an optional extra.** B2 targets are generated by
a multilingual LM from text, so on foreign audio some apparent multilingual
performance can come from the LM inferring intent from context and the 60-intent
prior rather than from the encoder conveying foreign phonetics. E1 bounds
exactly that. Without it, a good E2 number is unattributable. E1 doubles as the
pipeline check: if `FT_EN` + trained `proj` does not at least match `FT_EN` +
original `proj` on English, the training loop is broken.

**E2 is expected to be strong, possibly the strongest.** It is SLAM-ASR — a
frozen speech encoder, a frozen LLM, one trained linear projector — which
`LITERATURE.md` already lists as the honest upper baseline for whether the
interface is recoverable at all. Structurally it **dissolves the Pareto tradeoff
the repository has been fighting**: `PT_ML` stays byte-identical, so
multilingual retention is exactly 100% by construction, and interface
compatibility is bought from a separate 4.59M-parameter budget instead of from
`PT_ML`'s acoustics. It also avoids an ambiguity the merge arms carry: a merged
encoder has no canonical attention left context, so under invariant 5 E3/E4
collect `F`'s Gram contribution through a merged model running `PT_ML`'s
56-frame context, a setting `F` never saw. E2 has no such problem.

E2 is initialised from comparison 3's folded map rather than the original
`proj`, which makes it a clean single-variable comparison against work already
done: same encoder, same folded-map structure, objective changed from embedding
MSE to text CE. Comparison 3 reached English R² = −0.024 — still not beating a
mean predictor — yet recovered one of six calls, which is itself evidence that
R² was measuring the wrong thing. The LM does not need `PT_ML` frames mapped
onto `F`'s exact embeddings, only somewhere readable.

**Expect E2 to possibly win.** That is a real result, not a failure, and
`LITERATURE.md` §2.4 already predicts it.

### 9.1 The confound that must be controlled

The arms are **asymmetrically data-sensitive**. E3 starts near `F` and needs a
small correction with a good prior; E2 must learn a much larger re-basing from a
worse start. At a fixed data budget E3 is flattered, and the naive comparison
would report a merge advantage that is really a budget artifact.

Dataset B is ~8k clips at ~3 s ≈ **6.7 hours of unique audio**; SLAM-ASR trains
on ~960 h. Six hours may be ample for E3 and badly short for E2.

**Therefore compare scaling curves, not endpoints:** run each arm at 25/50/100%
of B. If E2 is still climbing at 100% while E3 has plateaued, the difference is
real rather than a measurement of the data budget.

If E2 is data-starved, the remedy is more data, not a better merge: add a
transcription auxiliary term (target = transcript) on Common Voice / MLS
fr-de-ru. That is what SLAM-ASR trains on, it is the interface-learning signal
E2 needs, and it scales without any teacher-generation step.

---

## 10. Hardware and precision

Fits a single RTX 3090 (24 GB). Three properties do most of the work:

- **No optimizer state for the LM.** Frozen, so no weight gradients and no Adam
  moments for ~10.4B parameters. `proj` costs ~74 MB for param+grad+m+v in fp32.
- **The encoder leaves the training loop.** After the merge it is fixed, so
  `proj`'s input is deterministic: precompute once, ~620 MB for 8k clips
  (38 frames × 1024 × 2 B). `asr_align/final_map.py` already has the frozen
  activation-cache reader.
- **The system prompt is a cacheable prefix.** Frozen, identical per condition,
  and positioned before the audio, so its KV never depends on `proj` and needs
  no gradient. Computing it once removes roughly two thirds of both the forward
  and the backward if the tool-definition prompt is ~400 of ~600 positions.

What remains is ~100--200 gradient-bearing positions per example (a 3 s command
is ~38 audio positions at 12.5 fps; the tool call is 30--80 tokens).

| | bf16 | INT8 | NF4 |
|---|---:|---:|---:|
| LM weights (~10.4B) | ~21 GB | ~11 GB | ~6 GB |
| cached prompt KV (estimate) | ~0.07 GB | ~0.07 GB | ~0.07 GB |
| `proj` + Adam (fp32) | ~0.07 GB | ~0.07 GB | ~0.07 GB |
| activations, batch 8 × 200 tok, checkpointed | ~1--2 GB | ~1--2 GB | ~1--2 GB |
| **total** | **~23 GB — OOM in practice** | **~13 GB** | **~8 GB** |

bf16 is the one configuration that does not fit. Start with NF4: roughly
16.6 TFLOPs per example, so ~3--6 hours for two epochs over ~8k examples at a
realistic 15--25 effective TFLOPS. Treat that as order-of-magnitude;
bitsandbytes dequant overhead is the main uncertainty and `LLM.int8()` in
particular is materially slower than NF4.

RegMean++ itself is a non-issue: \(L\) sequential forward passes over 400--700
clips through a 0.6B encoder is minutes.

**The real risk is precision coupling, not memory.** A `proj` fitted against an
NF4 LM partially absorbs NF4's quantization error, while deployment runs Q8_0
GGUF. Invariant 3's discipline needs extending: here the LM precision used for
*fitting* is itself an experimental variable, not only an export stage. Fit a
few hundred examples against NF4 and again against bf16 with CPU offload, compare
the resulting maps and pilot scores, and record the precision in each candidate's
provenance. INT8 at ~13 GB is the closer-to-Q8_0 fallback if the gap is large.

**Second node (3080 Ti mobile, 16 GB): not required.** It holds the NF4 LM
fine, so it would be a data-parallel peer, not a model shard. DDP is unusually
cheap here — the all-reduce payload is `proj`'s gradients only, 4.59M × 4 B =
18.4 MB per step, ~150 ms over 1 GbE against a ~1 s step — purely because the LM
is frozen. Still not worth it: a thermally-limited mobile 3080 Ti is roughly half
a 3090, so it buys ~1.4× for cross-node launch and sync complexity on a 3--6 hour
job. Shorten the frozen prompt before adding hardware.

---

## 11. Blocking checks, to run before building anything

**Run. Results in `COMPARISON_7_GATE_RESULTS.md`; comparison 7 is unblocked.**
This section keeps its original text below, because what the checks were asked
is part of the record; what they answered is summarised first.

- **Condition B has a teacher, and only one.** Through the checkpoint's
  inherited Nemotron-H chat format the model answers fr/de/ru in the language
  it was addressed in on 0.90/0.85/0.82 of utterances after the §11.4 usability
  gate, 0.95/0.90/0.94 before it. Through the deployment
  runtime's own perception-channel text path it does not: Russian collapses to
  0.07, answering in English 73% of the time and transliterating when it does
  not. Same weights, same sentence — the input mode, not the model. B2 targets
  must therefore be generated offline through the chat format.
- **Condition A disobeys rather than fails.** It answers in the input language
  on 15% of German and 9% of French utterances, never on Russian. §11.4's
  output-language gate removes exactly those, at a cost of 0–16% of condition
  A's foreign targets and 10–18% of condition B's.
- **`n_ff` = `intermediate_size` = 4096 = 4·`n_embd`, but it is not the widest
  linear.** `subsampling.linear` is, at d_in = 4352, so Dataset A needs
  **17,408 frames = 23.2 minutes per candidate** — §5's estimate, made exact.
- **B1 and B2 targets are not on one scale.** On 24 English clips the text
  target shares a median unigram F1 of 0.41 with what VoiceChat itself says on
  the same audio, and is a third longer. Equal term weights are not a
  defensible default.
- **Two constraints discovered on the way.** The pinned bridge's
  `render_system_prompt` strips non-ASCII, so no foreign-language prompt can
  reach the model through `/v1/realtime` at all; and the GGUF declares `</s>`
  as end-of-turn while the chat format ends on `<SPECIAL_12>`, so text use
  needs an explicit eos override or every reply runs to the token cap.

1. **Does the frozen LM read fr/de/ru text and answer in-language?** Run it
   text-only on ~100 MASSIVE utterances per language under both system prompts.
   Check that it produces sensible answers from native text; that under prompt B
   it replies in the input language; that under prompt A it reliably replies in
   English. **If prompt B fails, condition B has no teacher.** A capability the
   teacher lacks cannot be distilled, and the honest response is to report that
   rather than train against noise. Twenty minutes; decides whether half of
   comparison 7 exists.
2. **Confirm `n_ff` = `intermediate_size` from the config** before sizing
   Dataset A (§5).
3. **Text-path vs audio-path target gap.** Generate both on a small English
   subset and measure. B1 and B2 targets are produced by different procedures,
   so without this their losses are not on the same scale and the term weighting
   is arbitrary.
4. **Teacher quality gate.** If the LM's native-language generation is
   grammatical but weak, gate targets on output-language ID and report teacher
   quality as its own number, so the distillation ceiling is known.

### 11.1 What §6 and §8 must now say

§6's B2 line — "the frozen LM run text-only on the transcript" — was ambiguous
between two input modes and is resolved: **the chat format**, not the
perception channel. §8's blocking prerequisite is discharged. The teacher path
joins the initialization, training manifest, frozen system prompt and fitting
precision that invariant 6 already requires a gradient-fitted projection to
record.

---

## 12. Evaluation

Comparison 7 yields a 2×2 readout per arm:

| input | prompt A (English only) | prompt B (input language) |
|---|---|---|
| English | interface anchor — must not regress | control: should equal the cell to its left |
| fr/de/ru | cross-lingual content transfer | full multilingual capability |

**Invariant 9 makes these two comparison rows, not one table with a prompt
column.** Rows are comparable only under one frozen system prompt among other
things, and `build_comparison()` enforces it.

New automatic metrics become available: MASSIVE ships intent and slot labels for
every utterance and Speech-MASSIVE inherits them, so the held-out Speech-MASSIVE
`test` split gives content scoring over thousands of utterances on two separable
axes — output-language ID (cheap, discrete, the direct readout of whether
language identity survived) and intent/slot correctness. This is a considerably
stronger signal than the six-clip pilot, which remains the deployment check
under invariant 8.

---

## 13. Data hygiene

| Pool | Use |
|---|---|
| SLURP `train` | Dataset A, \(G_F\) |
| SLURP `val` | Dataset B, B1 |
| SLURP `test` | reserved |
| Speech-MASSIVE `dev` (first N) | Dataset A, \(G_M\) |
| Speech-MASSIVE `dev` (rest) | Dataset B, B2 |
| Speech-MASSIVE `test` fr/de/ru | held-out report |
| CoVoST 2 / Common Voice fr-de-ru | §6 OOD ablation only |
| FLEURS | untouched — evaluation only, per the data rules in `AGENTS.md` |
| recorded tool-calling pilot | untouched — invariant 8 |

FLEURS is the invariant most at risk here, because it is already manifested and
is the obvious wrong choice for Gram collection. CoVoST 2 / Common Voice derive
from Common Voice and FLEURS from FLoRes, so the ablation arm does not
contaminate the retention metric either.

Both new manifests go through `manifests.write_frozen` with the same
content-addressing and per-file SHA-256 verification as the LibriSpeech and
FLEURS manifests.

---

## 14. Invariant impact

- **Invariant 6** has been extended to authorise a projection fitted by gradient
  descent through the frozen language model, alongside comparison 3's
  closed-form map. E0 was already inside it as written. The extension requires a
  gradient-fitted projection to record its initialization, training manifest,
  frozen system prompt and fitting precision, and to leave every `encoder.*`
  tensor byte-identical to its arm's source; nothing else in the served graph may
  be trained. Comparison 7 is therefore no longer blocked on governance, only on
  the gating check in §11.
- **Invariant 3** has been extended so that the LM precision used for *fitting*
  is recorded in provenance and measured separately from the export quantization
  stage (§10).
- **Invariant 7** would need extending only if the per-candidate Gram weighting
  of §2 is used.
- **Invariant 9** is respected by treating each system prompt as its own
  comparison row (§12).
- **Invariant 5** is respected: merged candidates inherit `PT_ML`'s complete
  runtime configuration, including its 56-frame left context. The consequence
  for E3/E4 is recorded in §9.
- **Invariants 1, 2, 4** are untouched. `proj` never enters a task vector.

---

## 15. Corrections to `LITERATURE.md`

§3.3 states that RegMean++ "needs the activations comparison 1 already shards
under `activations/`, plus Gram matrices, and nothing else." That is true of
RegMean and **false of RegMean++**. The cached shards are the *candidates'* own
forward passes; ++ requires activations from the partially merged model,
recomputed at each depth. The paper is explicit: *"RegMean++ incurs additional
forward passes in the merged model to collect the inner-product matrices, yet
the merging time equals that of RegMean."* Budget \(L\) sequential passes over
the Gram set, not one cached sweep. Corrected in place.

---

## 16. References

| Item | Where |
|---|---|
| RegMean, ICLR 2023 | <https://arxiv.org/abs/2212.09849> |
| RegMean++, TMLR 2026 | `papers/2508.03121v3.pdf`, <https://arxiv.org/abs/2508.03121> |
| RegMean++ code | <https://github.com/nthehai01/RegMean-plusplus> |
| SLAM-ASR | <https://arxiv.org/abs/2402.08846> |
| REPAIR | <https://arxiv.org/abs/2211.08403> |
| SLURP (CC BY 4.0; 72k utts, 18 domains) | <https://arxiv.org/abs/2011.13205> |
| MASSIVE (1M, 51 langs, localisation of SLURP) | <https://arxiv.org/abs/2204.08582> |
| Speech-MASSIVE (12 langs; full train fr/de only) | <https://arxiv.org/abs/2408.03900> |
| CoVoST 2 (CC0; 21 X→En incl. fr/de/ru) | <https://github.com/facebookresearch/covost> |
