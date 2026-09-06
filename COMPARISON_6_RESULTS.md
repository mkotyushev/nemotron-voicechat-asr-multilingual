# Comparison 6: RegMean++ merge with the original projection

Measured on 2026-09-06. Closed-form merging buys the best English interface
agreement of any arm so far -- it is the first candidate whose VoiceChat-space
R² against `FT_EN` is positive -- and it is the first arm to pay for that in
measured multilingual retrieval. Comparison 3 bought a smaller English gain for
no retrieval cost; this arm buys a larger one and gives some back. Nothing here
is trained: every encoder tensor is a closed-form solve or an average, and
`proj` is the untouched `FT_EN` projection, so the candidate stays inside
invariant 6 as written.

## The candidate

`regmean-plus-plus`, RegMean++ (Nguyen et al., TMLR 2026, Algorithm 1) at
shrinkage α = 0.3. For each dense linear, with `X_i` the layer's input under
candidate `i`:

```text
W_M = [sum_i G_i]^-1 sum_i G_i W_i,   G_i = α X_i^T X_i + (1-α) diag(X_i^T X_i)
```

The cross-layer input is the **merged** prefix's output; only the intra-layer
sub-module inputs come from running each candidate's own layer on it. There is
no gradient descent and no ground truth anywhere in the merge; the only passes
over audio are forward passes, 25 of them in depth order.

`PT_EN` takes no part. RegMean combines full weights rather than deltas, so it
needs no shared base; `PT_EN` is loaded and validated only because the triplet
check is what enforces the exact-key, exact-shape invariant.

## Tensor routing

All 636 canonical `encoder.*` tensors are classified exactly once: **265
RegMean-solved, 371 averaged**. `conv.pointwise_conv{1,2}` are 1×1 convolutions
and are on the RegMean path, as the design record requires; `depthwise_conv`,
the subsampling convolutions, every LayerNorm, `bias_u`, `bias_v` and every bias
are averaged. Eq. 2 solves weight matrices only.

24 of the 265 solves collapse onto the unweighted mean, and they are exactly the
24 `self_attn.relative_k_proj` layers. Their input is the position encoding,
which does not depend on the audio, so the two candidates' Gram matrices are
identical and Eq. 2 reduces to the mean by construction. This is reported rather
than hidden: `distance_from_mean` is 0.0000 for all 24.

## Dataset A

Two different corpora, because equal Gram matrices would reduce Eq. 2 exactly to
the unweighted mean and erase everything RegMean adds over a soup:

| | source | clips | frames |
|---|---|---:|---:|
| `G_F` | SLURP `train`, English assistant commands | 447 | 17,433 |
| `G_M` | Speech-MASSIVE `validation` (dev) fr/de/ru, 149 each | 447 | 17,433 |

Both are fixed 3.0-second 16 kHz crops, so equalising frames across candidates
is exact rather than approximate and neither clip length nor collection volume
can act as an unintended merge coefficient. Each Gram is additionally normalised
by its own row count. 17,433 frames is **4.006 rows per input dimension** at the
widest linear, `subsampling.linear` (d_in = 4352) -- the budget
`COMPARISON_7_GATE_RESULTS.md` check 2 derived. FLEURS and LibriSpeech appear in
neither manifest.

Two selection facts worth stating rather than burying. 59% of SLURP takes are
shorter than 3 s and were skipped, so `G_F` is drawn from the longer half of the
corpus. The 100 MASSIVE utterance ids used by the comparison 7 gating check are
excluded from `G_M`.

## Shrinkage selection

α is selected on a held-out Dataset A split (112 clips per candidate, disjoint
from the Gram set) by agreement at the **encoder output** -- the only quantity
the frozen language model reads. The per-layer regression residuals are *not*
the criterion: RegMean++ solves each depth greedily against its own inputs and
never looks downstream. FLEURS and the speech-to-action pilot are not consulted.

| α | R² vs `PT_ML` (fr/de/ru) | R² vs `FT_EN` (en) | mean | weight L2 / `PT_ML` |
|---:|---:|---:|---:|---:|
| 0.1 | -4.950525 | -0.059848 | -2.505187 | 1.031 |
| **0.3** | **-4.947721** | **-0.028391** | **-2.488056** | 1.039 |
| 0.5 | -4.974966 | -0.008007 | -2.491486 | 1.048 |
| 0.7 | -5.041497 | +0.008391 | -2.516553 | 1.062 |
| 0.9 | -5.133726 | +0.022212 | -2.555757 | 1.094 |
| 0.95 | -5.146311 | +0.024340 | -2.560985 | 1.115 |

The two sides move in opposite directions monotonically and the symmetric
criterion peaks in the interior, at α = 0.3, rather than running to a grid edge.
α = 1.0 is excluded by the design record: it removes the shrinkage entirely.

## The solve is rank-deficient, and it had to be ridged

The unridged solve produced encoder weights **8.3 × 10¹⁴ times `PT_ML`'s norm** —
finite, plausible-looking, and built entirely out of round-off. Writing
`W_M = W̄ + D` and subtracting shows why: `D` is decided entirely by the
*difference* between the two Gram matrices, which is pure noise wherever the
encoder's activations do not reach, and the `(1-α) diag(G)` shrinkage vanishes in
exactly the directions that are dead in both candidates.

The solve is therefore for the offset from the candidates' mean, with a ridge at
`1e-6` of the regularized Gram's leading eigenvalue: directions the data
determines are unchanged, directions it does not stay at the mean. That is the
minimum-deviation choice among the objective's minimizers, not a different
objective, and it is recorded per layer. It brought the weight norm to
**1.039 × `PT_ML`** and left the held-out output agreement essentially unmoved
(α = 0.1: R² −4.9536/−0.0587 unridged against −4.9505/−0.0598 ridged), which is
the direct evidence that the blow-up lived in directions the data never visits.

How rank-deficient each layer is varies enormously, and the median is not the
problem:

| layer | determined directions | condition number |
|---|---|---|
| `subsampling.linear` | 3942 / 4352 | non-positive smallest eigenvalue |
| `layers.12.self_attn.q_proj` | 1024 / 1024 | 4.2 × 10³ |
| `layers.12.feed_forward1.linear2` | 225 / 4096 | 3.8 × 10¹³ |

Median across all 265 solves is 1.000 of the input width. The attention
projections and the subsampling stem are essentially fully determined; it is the
4096-wide FFN inner spaces, whose inputs are SiLU activations, that 17k frames
cannot resolve. More Gram audio would help those layers and nothing else.

## Shared evaluation

Against the exact frozen Comparison 1 arrays, same manifests, paired bootstrap.

| Metric | `PT_ML` pre | merge pre | `PT_ML` post Q8 | merge post Q8 |
|---|---:|---:|---:|---:|
| English VoiceChat-space R² | -0.700806 | **+0.002634** | -0.701194 | **+0.001729** |
| English VoiceChat-space cosine | 0.300820 | **0.570401** | 0.300750 | 0.570035 |

The pre-quantization paired R² improvement is **+0.703440**, 95% CI
[+0.6967, +0.7097]; cosine improves by **+0.269581**, CI [+0.2680, +0.2712].
This is the first candidate in the suite to reach R² > 0, i.e. to beat a mean
predictor of `FT_EN`'s embeddings. Comparison 3 reached −0.023870 and cosine
0.540775 on the same split; comparison 2 at λ=1 reached −0.752718.

Quantization costs R² 0.000905 and cosine 0.000366.

**The multilingual cost is real and is the headline limitation.** Historical
centered FLEURS retrieval degrades, and the French and German intervals exclude
zero:

| language | top-1 `PT_ML` | top-1 merge | difference | 95% CI |
|---|---:|---:|---:|---|
| fr | 0.646617 | 0.466165 | -0.180451 | [-0.2483, -0.1128] |
| de | 0.429577 | 0.309859 | -0.119718 | [-0.1901, -0.0563] |
| ru | 0.400000 | 0.351724 | -0.048276 | [-0.1241, +0.0276] |

Intrinsic candidate-to-candidate cross-lingual retrieval, measured before the
projection, is mostly preserved: French and Russian differences have intervals
containing zero on top-1 and MRR, German degrades (top-1 -0.077465,
[-0.1408, -0.0211]). Candidate-on-English retrieval improves slightly
(+0.0070 top-1 in all three groups).

This is the Pareto tradeoff comparison 3 avoided by never touching the encoder.
Comparison 6 moves 265 weight matrices and buys 0.70 of English R² for roughly a
tenth of FLEURS top-1. Comparison 2 at λ=1 destroys both.

## Arms

All at α = 0.3, held-out Dataset A, encoder output.

| arm | what it isolates | R² vs `PT_ML` | R² vs `FT_EN` | mean |
|---|---|---:|---:|---:|
| `regmean-plus-plus` | the comparison's candidate | -4.947721 | -0.028391 | -2.488056 |
| `regmean-plain` | the cross-layer correction | -5.586779 | +0.038983 | -2.773898 |
| `simple-average` | the Gram weighting; arm `E4`'s encoder | -4.686216 | -0.016802 | -2.351509 |
| `regmean-layernorm-ft-en` | LayerNorms seeded from `FT_EN` | -18.352554 | +0.470590 | -8.940982 |
| `regmean-ood-gram` | `G_M` re-collected off-domain | -4.521760 | -0.017727 | -2.269743 |

Three of these are negative results and should be read as such.

**Simple averaging beats RegMean++ on the selection criterion** (-2.351509
against -2.488056). On this encoder, at this data budget, the Gram weighting did
not earn its complexity by the measure used to select α. It is better on the
multilingual side and slightly worse on the English side.

**The off-domain Gram beats the in-domain one** (-2.269743, the best of all five
arms). Re-collecting `G_M` from Common Voice fr/de/ru read speech instead of
Speech-MASSIVE assistant commands *improved* held-out agreement, including
agreement with `PT_ML` measured on held-out Speech-MASSIVE. The paper's Table 5
ID-vs-OOD sensitivity does not replicate here in the direction predicted; the
most economical reading is that the Gram is capturing generic acoustics rather
than task structure, and that the narrower corpus generalises worse.

**Plain RegMean is worse than RegMean++** (-2.773898 against -2.488056), so the
cross-layer correction does what the paper claims on this model, even though the
Gram weighting as a whole does not beat the mean.

**Seeding LayerNorms from `FT_EN` is catastrophic for multilingual agreement**
(-18.35 against `PT_ML`) while improving English agreement to +0.47. Under ++ the
averaged non-linear tensors feed every downstream solve, and replacing them with
`FT_EN`'s pulls the whole merged model most of the way to `FT_EN`. It is a large
effect in a clearly interpretable direction, not a subtle one.

Only `regmean-plus-plus` was carried to the deployment stage. The other four
have full pre-quantization evaluation records and F32 artifacts;
`simple-average` is retained because Comparison 7 needs it as arm `E4`'s encoder.

## Paired speech-to-action evaluation

The five-row table is generated by `build_comparison()` and saved in
`.cache/experiments/voice-assistant-pilot-v2/analysis-through-comparison-6/`.
The frozen four-row table through comparison 3 is unchanged. All rows share the
same six semantic cases in English and Russian, the same frozen system prompt,
Q8 stage, runtime commit `229dc0e8`, runtime environment and 30-second response
budget measured from the end of each clip.

| Row | EN exact call | RU exact call | RU-EN paired | 95% CI |
|---|---:|---:|---:|---|
| control: original `FT_EN` encoder | 0.667 (4/6) | 0.000 | -0.667 | [-1.000, -0.333] |
| comparison 1, `PT_ML` baseline | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |
| comparison 2, `lambda=1` | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |
| comparison 3, final map | 0.167 (1/6) | 0.000 | -0.167 | [-0.500, +0.000] |
| **comparison 6, the merge** | **0.500 (3/6)** | 0.000 | -0.500 | [-0.833, -0.167] |

**This is the largest movement any arm has produced on the primary endpoint.**
The merge triples comparison 3's English exact-call rate and reaches three
quarters of what the original `FT_EN` encoder manages on the same six cases.
Where the failures now sit has also changed: comparison 6 produces an assistant
turn on 4 of 6 English clips and attempts a well-formed tool call on all four,
losing one to the tool name and one to an argument value. Comparisons 1 and 2
produced no turn at all; comparison 3 produced one.

Its three correct calls answer correctly in English: 120 for `math.factorial`,
`x = 1` and `x = 2` for the quadratic roots, and 6 square units for the triangle
area. A fourth clip attempted a call and produced the right spoken answer
(225 metres) with wrong structured arguments.

**Russian is 0/6 on every axis, including producing any turn at all.** All twelve
sessions completed with no transport errors, so this is the model, not the
harness. The `FT_EN` control answers all six Russian clips in English without
attempting a call; comparison 6 does not answer them at all. Improving the
English interface has not bought Russian tool calling, and the `Responded` column
separates the two failure modes.

This is a six-case development pilot with synthesized single-speaker Russian
audio, and it may be inspected before choices are made. It establishes neither
reliable English tool calling nor a deployable multilingual model.


## Reusable artifacts and verification

All paths are relative to this repository's checkout; large files stay ignored.

- `.cache/experiments/dataset-a-v1/`: the frozen SLURP and Speech-MASSIVE
  manifests, with archive hashes, extraction rules and a per-clip SHA-256, plus
  `dataset_a.json` recording the frame budget.
- `.cache/experiments/dataset-a-ood-v1/`: the Common Voice manifest for the
  off-domain Gram ablation, on the same budget.
- `.cache/experiments/comparison-6-regmean-v1/run.json`: command, checkpoint
  revisions and hashes, routing, Gram budget, alpha grid, arms and artifact index.
- Its `analysis/`: `alpha_grid.{json,md}`, `arms.{json,md}`,
  `merges/<arm>.json` (per-layer solves: condition number, effective rank,
  ridge, distance from the mean and from each candidate, output residual), and
  `delta-{pre,post}_quantization.{json,md}` against comparisons 1 and 2.
- Its `embeddings/<arm>/`: reusable evaluation arrays paired against the exact
  frozen Comparison 1 reference.
- Its `artifacts/<arm>/`: F32 encoder, the untouched `proj.*`, and featurizer
  tensors, so the deployment converter cannot fall back to the container's own
  projection.
- Its `deployment/regmean-plus-plus-Q8_0.gguf`: the evaluated deployment artifact.

Key SHA-256 values:

```text
shared_setup.json file:
37a872099a0b208a9ada1f6d752ddc53ab6ee31f73323f07370025769002cdcb
LibriSpeech manifest content:
a9bccd84c9696b842d49d1817920b10057eda9274315a4c2bef312bd1365b373
FLEURS manifest content:
9bba5fc8c05f398e08f8a14f3cbe19e7d0db88834895d01175e791b8a6db6af1
SLURP Dataset A manifest content:
bcadf68b6f17cb9737caab9fff6128f49f48d65580f703082d74022528d19764
Speech-MASSIVE Dataset A manifest content:
213912bbedb72d859f8cb06a8217dbee67695f1ed5f7050d3c0ac526842a40d9
Common Voice off-domain manifest content:
4ff508f02995f94a6c1253f51345c000ab2c1c42e3c0006619219f1dae3cf811
slurp_real.tar.gz (Zenodo md5 9efc0f058ced47bf5131c7cb2cade513):
9efc0f058ced47bf5131c7cb2cade513 (md5, as published)
exported model.safetensors (regmean-plus-plus):
0c2ce97e0cb9aa27e29c3193b1546fa8bbfcb623250bcac2362894bdd3c83c54
exported model.safetensors (simple-average):
413f93b1e7570be83680708bec2c7493ce0e8c57870a2146652c8302a81f954a
deployment GGUF:
60fbbd87acff91334aa8efd571bf4687c3d835e7fa0545c0d02fc4677ad45188
run.json:
63cd8d2040db1df0f4d02fb6730a1f1bcf7dbd7435ceb8726a3a4b64c0dcc93a
```

Upstream pins: SLURP metadata `pswietojanski/slurp@8eb16545762be97ace75334109d73824217311f1`,
Speech-MASSIVE `FBK-MT/Speech-MASSIVE@ff792febc16187a21e5bca38fb02a55daf91dc05`,
Common Voice 17.0 `fsicoli/common_voice_17_0@8262c16bf297c87a9cd88c51997c4758ed7a8ba2`.

Arithmetic ran in F32 with F64 solves on one CUDA device with deterministic
algorithms and TF32 disabled, torch 2.11.0+cu128, numpy 2.5.2, seed 0. Every
candidate inherits `PT_ML`'s complete runtime configuration including its
56-frame left context, which means `FT_EN`'s Gram contribution was collected
through a merged model running a context `FT_EN` never saw; the design record
records that consequence and the run record repeats it.

Deployment quantization changes 271 of 640 tensors, relative L2 0.005411,
maximum absolute change 0.247317. The actual Q8_0 artifact matches the in-memory
rounding model exactly.

## Limits

Per-layer regression residuals are not evidence of success. RegMean++ solves each
depth greedily and controls no error at the encoder output, and this run shows
concretely what that permits: solutions that fit every layer's own inputs well
while carrying an arbitrary null-space component, which only the ridge removed.

Retrieval remains screening evidence. The English R² gain is the largest in the
suite and the FLEURS retrieval loss is real, but neither says whether the frozen
language model can act on what it hears; only the speech-to-action row does, on
six single-call cases with synthesized Russian audio.

Comparison 7 remains unstarted. `simple-average` is exported and evaluated here
precisely because it is arm `E4`'s encoder there, and its result above is a
reason to take that arm seriously rather than treat it as a formality.
