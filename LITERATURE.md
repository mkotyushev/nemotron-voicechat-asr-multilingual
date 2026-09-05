# Literature review: combining two fine-tunes of one pre-trained encoder

This file surveys published, peer-reviewed, or community-adopted methods for the
problem this repository is built around, and maps each to the comparison slots in
`EXPERIMENTS_TODO.md`. It is a reading and planning document. It records no new
measurement and it does not check off any experiment box.

The problem, in this repository's notation:

```text
E = PT_EN   base pre-trained English streaming encoder
F = FT_EN   VoiceChat's encoder, a light fine-tune of E under joint training
M = PT_ML   multilingual encoder, a heavy continued pre-training of E
want FT_ML  a model with F's LLM-readable interface and M's multilingual acoustics
```

Optimal-transport and projection ideas already recorded in `README.md` and
comparisons 3--5 are treated as prior work here and are not restated, except
where a published method subsumes or contradicts them.

## 1. The problem already has a name

`C(lambda) = M + lambda * (F - E)` is the **chat vector** recipe. Huang et al.
compute a delta between an instruction-tuned model and its base, and add it to a
*continually pre-trained* model in a new language, obtaining instruction
following in that language without further training. That is structurally
identical to comparison 2, with `F - E` in the role of the chat vector and `M` in
the role of the continually pre-trained model.

- Chat Vector, ACL 2024 --- <https://arxiv.org/abs/2310.04799>, code at
  <https://github.com/aqweteddy/ChatVector>

This matters for two reasons. First, comparison 2 is not an exotic construction;
it is the standard recipe, so its negative result here is informative rather than
a sign of a coding error. Second, the recipe's known failure modes are documented
and are directly applicable, which is the subject of section 2.

Two further papers state the same shape in the cross-lingual setting:

- *The Unreasonable Effectiveness of Model Merging for Cross-Lingual Transfer in
  LLMs* --- <https://arxiv.org/abs/2505.18356>. Merges a task expert with a
  language-adapted (continually pre-trained) expert. Its headline methodological
  finding is that **layer-wise merging coefficients substantially beat a single
  uniform coefficient**, with the language expert favoured at different depths
  than the task expert.
- *Layer Swapping for Zero-Shot Cross-Lingual Transfer in LLMs*, ICLR 2025 ---
  <https://arxiv.org/abs/2410.01335>. From one base, fine-tune an English task
  expert and a target-language expert, then **replace the top and bottom
  transformer layers of the task expert with the language expert's layers**. No
  arithmetic at all. Reported +10% on MGSM across four languages over the
  individual experts and over other merging methods.

The multimodal analogue is closer still to this repository's setting:

- *DiM^3: Bridging Multilingual and Multimodal Models via Direction- and
  Magnitude-Aware Merging* --- <https://arxiv.org/abs/2605.12960>. Composes a
  multilingual continued-pre-training residual with a multimodal residual on a
  shared backbone, **while leaving the vision encoder and the multimodal
  projector untouched** --- the same constraint invariant 6 imposes on `proj.*`
  here.

## 2. Why the published diagnosis predicts our negative result

### 2.1 The two deltas are not the same size

DARE's central measurement is that supervised fine-tuning deltas are tiny
(values typically within 0.002) and 90--99% redundant, whereas **continued
pre-training produces much larger delta magnitudes, and methods calibrated on
SFT-scale deltas damage continually pre-trained models**.

- DARE / *Language Models are Super Mario*, ICML 2024 ---
  <https://arxiv.org/abs/2311.03099>

Here `M - E` is a continued-pre-training delta and `F - E` is a joint-training
delta. The weight cosines already recorded in `README.md` show the asymmetry
directly, and show that it grows with depth:

| Tensor | VC~EN (`F`~`E`) | EN~ML (`E`~`M`) |
|---|---:|---:|
| subsampling linear | 0.966 | 0.944 |
| first-block Q projection | 0.855 | 0.737 |
| block-12 V projection | 0.909 | 0.686 |
| block-23 FFN input | 0.934 | 0.621 |

Adding an unscaled `F - E` on top of `M` therefore perturbs a model that has
already moved much further from `E` than `F` has, and it perturbs it most where
`M` has moved most. A single global lambda cannot express that.

### 2.2 Same initialisation does not guarantee a shared basin

Fine-tunes of a shared initialisation usually stay linearly mode connected, which
is what makes task arithmetic work at all; extensive continued pre-training is
one of the documented conditions under which that connectivity degrades.

- *Model Merging in the Era of Large Language Models* (survey) ---
  <https://arxiv.org/abs/2603.09938>
- *Model Merging in LLMs, MLLMs, and Beyond*, ACM Computing Surveys 2026 ---
  <https://arxiv.org/abs/2408.07666>; curated index at
  <https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications>

### 2.3 Merged networks lose activation variance

Two papers give the same diagnosis for why a merged network can be
catastrophically worse than either parent while its weights look reasonable:
the interpolated network's **activation variance collapses**, and the fix is an
affine per-channel renormalisation, not a better weight combination.

- REPAIR, ICLR 2023 --- <https://arxiv.org/abs/2211.08403>. Rescaling
  pre-activations of interpolated networks removes 60--100% of the interpolation
  barrier across architectures.
- *Vanishing Feature: Diagnosing Model Merging and Beyond* ---
  <https://arxiv.org/abs/2402.05966>. Ties the failure to feature magnitude decay
  through depth, notes that it is worse when the merged models' weight magnitudes
  differ substantially, and that normalisation layers determine whether the
  network can recover on its own.

This is the most immediately testable hypothesis for the recorded end-to-end
failure, because the evaluator already emits embedding mean/norm diagnostics and
because a per-channel affine correction folds into `proj` by exactly the
mechanism `README.md` already documents for the interface map.

### 2.4 The interface, not the merge, is the currently binding constraint

The recorded speech-to-action pilot shows **no assistant turn for comparison 1 at
all** --- that is `lambda = 0`, the unmodified `M` with the unchanged VoiceChat
projection, no arithmetic involved. Whatever comparison 2's arithmetic does or
does not do, a merge cannot be evaluated end to end while the zero point of the
sweep is already mute. The literature in section 3.8 addresses this directly and
should be sequenced first.

### 2.5 The permutation result is expected, and correctly closes that door

Transport plans coming out as the identity, with cost essentially equal to
identity cost, is the predicted outcome for two models that share an
initialisation: they drift within a basin rather than renumbering neurons.
Permutation and optimal-transport alignment pay off when the models were trained
from *different* initialisations.

- Git Re-Basin --- <https://arxiv.org/abs/2209.04836>
- Model Fusion via Optimal Transport, NeurIPS 2020 ---
  <https://arxiv.org/abs/1910.05653>
- Transformer Fusion with Optimal Transport, ICLR 2024 ---
  <https://arxiv.org/abs/2310.05719>. Handles multi-head attention, LayerNorm and
  residuals, and finds soft (Sinkhorn) alignment consistently beats hard EMD for
  transformers. This is the correct reference if a soft version is ever wanted.
- *A correlation-permutation approach for speech-music encoders model merging* ---
  <https://arxiv.org/abs/2506.11403>. Permutation alignment paying off for two
  audio encoders --- but note that HuBERT and MERT do **not** share an
  initialisation, which is exactly why it works there and not here.

Recommendation: keep the transport code as the diagnostic it currently is, and
stop treating it as a candidate route for this model pair.

## 3. Method families worth trying, ranked by fit

### 3.1 Layer-wise coefficients instead of one global lambda

The cheapest substantive upgrade to comparison 2, and the one our own weight
cosines argue for most directly.

- **LiNeS**, ICLR 2025 --- <https://arxiv.org/abs/2410.17146>. Scales the task
  vector linearly with layer depth, keeping shallow layers near pre-trained
  values to preserve general features and letting deep layers carry the
  task-specific update. Designed exactly to trade forgetting against transfer,
  and reported to help both single-task and merging settings.
- **AdaMerging**, ICLR 2024 --- <https://arxiv.org/abs/2310.02575>. Learns
  task-wise or layer-wise coefficients with no labels, by minimising prediction
  entropy on unlabelled test data. Reported ~11% over task arithmetic. The
  unsupervised objective would need a speech-appropriate surrogate here (a CTC or
  next-token entropy on unlabelled audio); that is a real design question, not a
  drop-in.
- **Model Stock**, ECCV 2024 oral --- <https://arxiv.org/abs/2403.19522>, code at
  <https://github.com/naver-ai/model-stock>. Uses the pre-trained weight as an
  anchor and derives a **per-layer** merging ratio from the geometry of just two
  fine-tuned models. Two fine-tuned models from one base is precisely our
  configuration, and it needs no data and no tuning.

Suggested form for a comparison 2b: `C = M + diag(lambda_l) * (F - E)` with
`lambda_l` from (i) a LiNeS depth ramp and (ii) a Model Stock closed-form ratio,
evaluated on the existing screening metrics. Note that the fixed sweep in
`asr_align.experiments` and invariant 7 would need an explicit extension before
any such coefficient is introduced.

### 3.2 Composition at layer granularity, with no arithmetic

Layer Swapping (section 1) and ZipIt!'s partial merging both report that
composing whole layers beats blending every parameter.

- **ZipIt!**, ICLR 2024 --- <https://arxiv.org/abs/2305.03053>. Its two
  contributions are merging redundant features *within* a model, and **partial
  merging up to a chosen layer**, which alone "can improve accuracy by over 15%
  while still keeping most layers merged".

For a 24-block encoder feeding a frozen projection, the natural hypothesis is:
keep `M`'s acoustic front end (featurizer, subsampling, early blocks) and take
`F`'s late blocks, which are the ones that were shaped to be readable by the
VoiceChat projection --- and which are also where `E`~`M` cosine is worst. The
search space is small enough to grid: one integer split point, or two for a
top-and-bottom swap. No fitting, no data, no new invariants beyond the lambda
rule, and it is directly falsifiable against comparison 2's Pareto curve.

### 3.3 Closed-form activation-matched merging (supersedes the planned comparisons 4 and 5)

Comparisons 4 and 5 propose fitting bidirectional identity-regularised maps for
every residual boundary, FFN hidden space, convolution-channel space and
attention head, then transporting each delta as
`dW_M = A_out^T dW_E B_in^T`, with a hand-maintained routing manifest for every
structurally incompatible tensor. There is a published method that achieves the
same objective --- make the merged layer's *outputs* match --- in closed form and
with far less machinery:

- **RegMean**, ICLR 2023 --- <https://arxiv.org/abs/2212.09849>. For each linear
  layer, solve a least-squares problem that minimises the distance between the
  merged layer's output activations and each source layer's outputs. The solution
  needs only the Gram (inner-product) matrices of each layer's input activations,
  collected by forward passes on unlabelled data.
- **RegMean++**, TMLR 2026 --- <https://arxiv.org/abs/2508.03121>, code at
  <https://github.com/nthehai01/RegMean-plusplus>. Fixes RegMean's known flaw:
  merging each linear layer independently ignores how earlier merged layers
  change the inputs of later ones. RegMean++ feeds the *merged* model's own
  activations forward layer by layer.

Why this is a better use of the same effort:

- It needs the activations comparison 1 already shards under `activations/`,
  plus Gram matrices, and nothing else.
- It never constructs an unsupported dense operation, so the LayerNorm,
  depthwise-convolution, GLU and attention-structure special cases that
  comparisons 4 and 5 exist to arbitrate simply do not arise; non-linear tensors
  are left to a simple rule and reported as such.
- RegMean++'s cross-layer correction is the precise failure that comparison 4's
  layerwise transported-update agreement test is designed to detect. Better to
  adopt the method that already fixes it.
- It is a merge, not a transport of a delta, so it is evaluated on the same
  screening metrics with no new result schema.

The bespoke transport plan retains one advantage worth keeping: it produces a
per-tensor classification manifest. That is good practice and should survive into
whatever replaces it.

### 3.4 Spectral and subspace interference control

- **TSV-M** (Task Singular Vectors), CVPR 2025 ---
  <https://arxiv.org/abs/2412.00081>, code at
  <https://github.com/AntoAndGar/task_singular_vectors>. SVD each layer's task
  matrix, keep dominant directions, orthogonalise across tasks, recombine.
- **Iso-C / Iso-CTS**, ICLR 2025 --- <https://arxiv.org/abs/2502.04959>.
  Isotropic merging: flatten the singular-value spectrum of the merged update,
  with common and task-specific subspaces.
- **TIES-Merging**, NeurIPS 2023 --- <https://arxiv.org/abs/2306.01708>. Trim,
  elect sign, disjoint merge. The standard first sparsity baseline.
- **Model Breadcrumbs**, ECCV 2024 --- <https://arxiv.org/abs/2312.06795>.
  Sparsifies by removing *both* outliers and negligible values.

The single most transferable evidence for this repository comes from the speech
version of this benchmark:

- *Exploring the potential and limitations of Model Merging for Multi-Domain
  Adaptation in ASR* --- <https://arxiv.org/abs/2603.05354>. Eleven merging
  algorithms on speech-encoder domain adaptation, in three families: parameter
  space (Model Soups, Model Stock, Karcher mean, Multi-SLERP), task-vector space
  (Task Arithmetic, TIES, PCB-Merging, Select-Calculate-Erase), and task-vector
  subspace (Iso-C, Iso-CTS, TSV-M). They add **BoostedTSV-M**, which clamps every
  singular value below the one at a cumulative-energy threshold `beta` up to that
  value before concatenation, mitigating rank collapse; `beta = 0.3` was best over
  a `[0.1, 1.0]` sweep.

Its result splits exactly along this repository's Pareto axis: **task-vector
subspace methods won in-domain, while parameter-space methods preserved
cross-lingual ability best** --- Model Stock attained the lowest average FLEURS
error rate (6.79%), and Karcher mean the best target-language result. FLEURS is
the same benchmark used in `crosslingual_probe.py`. If the retention side of our
Pareto curve is the binding constraint, that paper says to look at Model Stock,
Karcher mean and Multi-SLERP before looking at sparsity methods.

Other speech-domain merging results, for context and for baselines:

- *Speech-FT*, TASLP 2026 --- <https://arxiv.org/abs/2502.12672>, code at
  <https://github.com/nervjack2/Speech-FT>. Fine-tune with reduced
  representational drift, then interpolate back toward the pre-trained model to
  restore cross-task generalisation. Relevant if we ever fine-tune anything
  ourselves rather than only merging published checkpoints.
- *Selective Attention Merging* --- <https://arxiv.org/abs/2501.08468>. Merges
  task vectors from **attention matrices only**; up to 14% relative WER reduction
  in a low-resource setting. A cheap ablation of comparison 2: restrict the task
  vector to attention tensors.
- *LoRS-Merging*, low-rank plus sparse merging for multilingual ASR and ST ---
  <https://arxiv.org/abs/2502.17380>.
- *Distilling a speech and music encoder with task arithmetic*, Interspeech 2025
  --- <https://arxiv.org/abs/2505.13270>.

### 3.5 Transporting a delta into the target model's own subspace

This is the reviewed form of the projection idea, and it is data-free:

- **LoRA-X**, ICLR 2025 --- <https://arxiv.org/abs/2501.16559>. Transfers an
  adapter across base models by **constraining it to a subspace of the target
  model's weights**, matching layers by a subspace-similarity metric and scoring
  transfer feasibility with an optimal-transport cost. The ablation reports that
  performance drops sharply without the subspace projection.
- **Cross-LoRA** --- <https://arxiv.org/abs/2508.05232>. LoRA-Align does
  rank-truncated SVD plus a Frobenius-optimal linear transform between the two
  bases; LoRA-Shift projects the source update into the target parameter space.
  Data-free.

Applied full-rank here: for each `encoder.*` matrix, take the SVD of `W_M`,
project `F - E` onto the span of its leading left and right singular vectors, and
add only that component. The discarded component is the part of the English
fine-tune that has no support in the multilingual model's own weight geometry ---
which is a defensible reason to drop it, and a quantity worth reporting per
tensor.

### 3.6 Direction- and magnitude-aware composition of heterogeneous updates

DiM^3 (<https://arxiv.org/abs/2605.12960>) is the closest published match to this
repository's exact problem statement. Its rule, per column `j` of each matrix:

```text
W~[:,j] = W_base[:,j] + sum_k omega[k,j] * Delta_k[:,j],   k in {multilingual, multimodal}
omega[k,j] = ( s_mag[k,j] + s_dir[k,j] ) / 2
```

with a DoRA-style magnitude/direction split of each column, a direction deviation
`1 - cos(D_k[:,j], D_base[:,j])`, a magnitude deviation `|m_k(j) - m_base(j)|`,
and both converted to salience by **rank-based** normalisation followed by
softmax across sources --- the rank step being what removes the
source-and-module-specific scale variation that defeats naive addition when one
update is a continued-pre-training update and the other is not. Reported to beat
Task Arithmetic, DARE, TIES, Breadcrumbs and PCB-Merging over 57 languages, while
leaving the vision encoder and projector untouched.

Note the structural difference to adapt: DiM^3 has a common base plus two
residuals in a *shared LLM backbone*, whereas here the two residuals live in the
encoder and the consumer is a frozen projection. The per-column rank-normalised
salience rule transfers; the surrounding architecture does not.

### 3.7 Repair the merged model's statistics after merging

REPAIR (section 2.3) computes per-channel affine rescale-and-shift coefficients
so that the merged network's channel statistics match a reference. Two properties
make it attractive here:

- It is orthogonal to the choice of merging method, so it composes with anything
  in sections 3.1--3.6.
- At the final layer it folds into `proj` by the mechanism already implemented in
  `asr_align/interface.py`, with no runtime graph change. Inside the encoder it
  folds into the existing LayerNorm scale and bias, which is also free.

The corresponding diagnostic --- per-channel mean and variance of each candidate's
hidden states against `F`'s on the same English audio --- costs one pass over
already-cached activations and would either explain or rule out the mute
end-to-end result before any further merging work is done.

### 3.8 Change the interface instead of the weights

The single English answer that the fitted map restored, and the fact that the
`lambda = 0` row is mute, both point here.

- **Model stitching**, NeurIPS 2021 --- <https://arxiv.org/abs/2106.07682>. The
  canonical framing of comparison 3: freeze two models, connect the bottom of one
  to the top of the other with a single **trained** layer. The literature's
  stitching layer is trained by gradient descent on a task loss; comparison 3's
  ridge fit is a data-free approximation of it against frozen activation pairs,
  and should be expected to underperform a trained stitch. See also
  <https://arxiv.org/abs/2303.11277> for what stitching does and does not measure.
- **SLAM-ASR**, <https://arxiv.org/abs/2402.08846>. A frozen speech encoder, a
  frozen LLM, and **only a trainable linear projector** reaches the best
  LibriSpeech result among LLM-based ASR systems. This is the honest upper
  baseline for everything in this repository: if a trained projector on modest
  paired data recovers the interface, then any training-free merge must be judged
  against it, not against the mute `PT_ML` row. It is out of scope under
  invariant 6 as currently written and would need a new comparison slot.
- **LegoSLM**, EMNLP Findings 2025 --- <https://arxiv.org/abs/2505.11352>. Trains
  the speech encoder to emit CTC posteriors **over the LLM's vocabulary**, then
  reconstructs pseudo-audio embeddings as a posterior-weighted sum of the LLM's
  input embeddings. Because the interface is text-space rather than a private
  1024-d latent, encoders become hot-swappable. Both NVIDIA checkpoints here
  already have CTC/decoder heads, which makes this unusually feasible --- but it
  is an architecture change to the served graph, not a merge.
- Projector transfer across languages, and mixture-of-experts projectors for
  massively multilingual settings, are surveyed in
  <https://arxiv.org/abs/2603.07025> and <https://arxiv.org/abs/2601.19451>.

### 3.9 Data-dependent weighting, if a small amount of data is acceptable

- **Fisher-weighted averaging**, NeurIPS 2022 --- <https://arxiv.org/abs/2111.09832>.
  Weight each parameter by its Fisher information, i.e. by how much each model
  actually depends on it.
- *Model Merging with Functional Dual Anchors* --- <https://arxiv.org/abs/2510.21223>.
- *Sens-Merging*, sensitivity-guided balancing --- <https://arxiv.org/abs/2502.12420>.

## 4. What the literature says to stop doing

| Currently in the plan | Why to drop or demote |
|---|---|
| Optimal transport / permutation alignment as a fusion route | Measured identity here, and predicted to be identity for two fine-tunes of one initialisation. Keep as a diagnostic only. |
| DARE-style random drop at high ratios | Documented to be damaging exactly for continued-pre-training-scale deltas, which is what `M - E` is. |
| A single global lambda | Dominated by layer-wise coefficients in both the cross-lingual merging study and LiNeS/Model Stock, and contradicted by our own depth-increasing drift. |
| Bespoke bidirectional activation maps for every internal space (comparisons 4 and 5) | RegMean/RegMean++ achieve the same output-matching objective in closed form, with fewer moving parts and no structured-tensor routing problem. |

## 5. Suggested order of work

Cheapest and most diagnostic first. Each step is falsifiable against results the
repository already has.

1. **Variance/statistics diagnostic (REPAIR).** Per-channel mean and variance of
   each existing candidate's hidden states versus `F` on the same English audio,
   from cached activations. Either explains the mute rows or rules the hypothesis
   out. Cost: one pass, no new artifacts.
2. **Layer-wise lambda (comparison 2b).** LiNeS depth ramp and Model Stock's
   closed-form per-layer ratio, on the existing screening metrics. Requires an
   explicit extension of the fixed lambda sweep and invariant 7.
3. **Layer swapping / partial merge (comparison 2c).** Grid over one or two split
   points. No fitting, no data.
4. **Replace comparisons 4 and 5 with RegMean++ (comparison 4').** Reuses the
   activation shards already produced by comparison 1. Keep the per-tensor
   classification manifest from the original plan.
5. **Parameter-space and subspace merges for the retention side.** Model Stock,
   Karcher mean, Multi-SLERP, then BoostedTSV-M / Iso-C, guided by the ASR
   benchmark's split between in-domain and FLEURS retention.
6. **If encoder-space merging still does not produce an assistant turn:** a
   trained stitch, i.e. SLAM-ASR-style projector training, as the upper baseline
   that says whether the interface is recoverable at all; and LegoSLM as the
   architecture-level answer to encoder swapping. Both need new comparison slots
   and both touch invariant 6.

Steps 1--4 are training-free and stay inside the existing evaluation contract.
Step 6 does not, and should not be started before steps 1--3 have said whether
the interface or the merge is the binding constraint.

## 6. Index

| Method | Venue | Link |
|---|---|---|
| Chat Vector | ACL 2024 | <https://arxiv.org/abs/2310.04799> |
| Task Arithmetic | ICLR 2023 | <https://arxiv.org/abs/2212.04089> |
| TIES-Merging | NeurIPS 2023 | <https://arxiv.org/abs/2306.01708> |
| DARE / Super Mario | ICML 2024 | <https://arxiv.org/abs/2311.03099> |
| Model Breadcrumbs | ECCV 2024 | <https://arxiv.org/abs/2312.06795> |
| RegMean | ICLR 2023 | <https://arxiv.org/abs/2212.09849> |
| RegMean++ | TMLR 2026 | <https://arxiv.org/abs/2508.03121> |
| Fisher-weighted averaging | NeurIPS 2022 | <https://arxiv.org/abs/2111.09832> |
| AdaMerging | ICLR 2024 | <https://arxiv.org/abs/2310.02575> |
| Model Stock | ECCV 2024 | <https://arxiv.org/abs/2403.19522> |
| LiNeS | ICLR 2025 | <https://arxiv.org/abs/2410.17146> |
| Layer Swapping | ICLR 2025 | <https://arxiv.org/abs/2410.01335> |
| ZipIt! | ICLR 2024 | <https://arxiv.org/abs/2305.03053> |
| TSV-M | CVPR 2025 | <https://arxiv.org/abs/2412.00081> |
| Iso-C / Iso-CTS | ICLR 2025 | <https://arxiv.org/abs/2502.04959> |
| LoRA-X | ICLR 2025 | <https://arxiv.org/abs/2501.16559> |
| Cross-LoRA | preprint | <https://arxiv.org/abs/2508.05232> |
| DiM^3 | preprint | <https://arxiv.org/abs/2605.12960> |
| REPAIR | ICLR 2023 | <https://arxiv.org/abs/2211.08403> |
| Vanishing Feature | preprint | <https://arxiv.org/abs/2402.05966> |
| Git Re-Basin | ICLR 2023 | <https://arxiv.org/abs/2209.04836> |
| Model Fusion via OT | NeurIPS 2020 | <https://arxiv.org/abs/1910.05653> |
| Transformer Fusion with OT | ICLR 2024 | <https://arxiv.org/abs/2310.05719> |
| Model stitching | NeurIPS 2021 | <https://arxiv.org/abs/2106.07682> |
| SLAM-ASR | preprint | <https://arxiv.org/abs/2402.08846> |
| LegoSLM | EMNLP Findings 2025 | <https://arxiv.org/abs/2505.11352> |
| Speech-FT | TASLP 2026 | <https://arxiv.org/abs/2502.12672> |
| Selective Attention Merging | preprint | <https://arxiv.org/abs/2501.08468> |
| LoRS-Merging | preprint | <https://arxiv.org/abs/2502.17380> |
| Speech/music correlation-permutation merging | preprint | <https://arxiv.org/abs/2506.11403> |
| Cross-lingual merging effectiveness | preprint | <https://arxiv.org/abs/2505.18356> |
| Model merging for multi-domain ASR (11 algorithms, BoostedTSV-M) | preprint | <https://arxiv.org/abs/2603.05354> |
| Model merging survey | ACM CSUR 2026 | <https://arxiv.org/abs/2408.07666> |
