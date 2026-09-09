# Repository guide for coding agents

Read this file, `EXPERIMENTS_TODO.md`, and the relevant CLI/module before
changing code. `EXPERIMENTS_TODO.md` is the current experiment specification;
complete and check off one comparison at a time. A checked implementation item
does not imply that a model experiment has been run.

`LITERATURE.md` surveys the published methods for this problem and maps them onto
the comparison slots. It is a planning document only: it never supersedes
`EXPERIMENTS_TODO.md`, and naming a method there does not authorise changing a
fixed sweep or an invariant without a decision recorded in this file.

`REGMEAN_INTERFACE_DESIGN.md` is the design record behind comparisons 6 and 7:
the merging objective, the rejected alternatives, the dataset composition, and
the blocking checks. Read it before implementing either. It records no
measurement except the §11 gating checks, whose results are indexed in
`COMPARISON_7_GATE_RESULTS.md`; invariant 6 already authorises comparison 7's
projection, so that comparison is no longer blocked.

## What this repository is

This is the research and checkpoint-export repository for replacing the
English perception encoder in NVIDIA NemotronLabs VoiceChat 11B with NVIDIA's
multilingual streaming ASR encoder. It is not the serving repository and it is
not evidence of a deployable multilingual VoiceChat model.

The sibling `nemotron-voicechat-realtime-gguf` repository consumes exported
checkpoints. Keep experimental fitting, probes, tensor arithmetic, and research
claims here. Do not edit or invoke the sibling repository unless the user
explicitly puts it in scope.

## Current experiment definition

The three fixed checkpoint roles are:

```text
E = PT_EN = nvidia/nemotron-speech-streaming-en-0.6b
M = PT_ML = nvidia/nemotron-3.5-asr-streaming-0.6b
F = FT_EN = nvidia/NVIDIA-NemotronLabs-VoiceChat-11B perception encoder
```

The direct task-arithmetic arm is:

```text
delta_F = F - E
C(lambda) = M + lambda * delta_F
lambda in {0, .25, .5, .75, 1}; lambda=1 is the primary endpoint
```

This definition supersedes the older explanatory direction in the top-level
docstring of `asr_align/fuse.py` (`VC + lambda * (ML - EN)`). The reusable
transport/rebase code in that module is still relevant, but new comparison code
must follow `EXPERIMENTS_TODO.md` and `asr_align/experiments.py`.

Shared setup and Comparisons 1--3 and 6 have validated artifacts and metrics.
Their completion evidence is indexed in `COMPARISON_3_RESULTS.md` and
`COMPARISON_6_RESULTS.md`. Comparisons 4, 5, 7 and the final comparison remain
experimental work; do not mark their result-oriented boxes complete without
producing and validating the stated artifacts and metrics. Completion is not a
claim of deployment quality.

Comparison 6 needed no invariant change and made none: `proj` is the untouched
`FT_EN` projection and nothing is trained. It is currently the strongest arm on
both the English interface metric and the speech-to-action endpoint, and the
first to pay a measured multilingual cost. Two of its recorded results should
shape what comes next rather than be rediscovered: **simple averaging beat
RegMean++ on the selection criterion**, and an **off-domain Common Voice Gram
beat the in-domain Speech-MASSIVE one**, which is the opposite of the paper's
Table 5 direction. Comparison 7's arm `E4` is therefore not a formality.

Comparison 7 is the only arm that trains anything; invariant 6 authorises it and
the language-model gating check in `REGMEAN_INTERFACE_DESIGN.md` §11 has been
run, so it is unblocked. That check settled one thing the comparison must honour: the frozen
language model answers fr/de/ru in-language through its inherited chat format
but not through the deployment runtime's perception-channel text path, so
condition B's targets are generated through the chat format and the teacher path
is recorded in provenance. `COMPARISON_7_GATE_RESULTS.md` has the numbers.

## Source map

- `shared_setup.py`: validates pinned checkpoint identities and lineage,
  records files/configurations, validates the arithmetic triplet, and writes
  immutable data manifests plus `shared_setup.json`.
- `shared_setup.example.json`: template for local paths, immutable revisions,
  lineage evidence, precision policy, and dataset roots.
- `asr_align/experiments.py`: authoritative roles, fixed lambda sweep, strict
  canonical tensor validation, F32 arithmetic, provenance hashes, and exact
  PT_ML runtime-config inheritance.
- `asr_align/manifests.py`: deterministic, content-addressed LibriSpeech and
  FLEURS manifest creation and audio-file verification.
- `asr_align/evaluation.py`: the versioned result contract and common evaluator
  for every comparison, including paired bootstrap intervals against PT_ML.
- `voice_assistant_evaluation.py` and `asr_align/voice_assistant.py`: the paired
  English/Russian speech-to-action evaluation. Every candidate is served by the
  pinned deployment runtime and scored before TTS on its structured tool call
  and assistant text.
- `direct_task_arithmetic.py` and `asr_align/direct.py`: the Comparison 2 runner,
  task-vector norm/reconstruction reports, frozen-baseline validation,
  activation-growth checks, and development Pareto table.
- `final_map_projection.py` and `asr_align/final_map.py`: the Comparison 3
  runner, identity-regularized bidirectional final-activation ridge maps,
  held-out regularization selection, map conditioning/cycle-consistency
  reports, the FLEURS generalization test, projection folding and its
  verification, the byte-identity check on `encoder.*`, and the paired table
  against Comparison 1. The evaluation passes are imported from the
  Comparison 2 runner so both arms produce rows with the same code.
- `asr_align/weights.py`: maps ASR safetensors, the original VoiceChat
  safetensors, and deployment GGUFs onto one canonical state-dict naming scheme.
- `asr_align/encoder.py` and `asr_align/features.py`: PyTorch port of the exact
  deployed FastConformer graph and featurizer.
- `asr_align/interface.py`: interface-map fitting, projection folding, and
  held-out VoiceChat-space scoring.
- `asr_align/hooks.py` and `asr_align/transport.py`: activation collection and
  transport-plan solving.
- `asr_align/fuse.py`: weight rebasing plus the strict task-arithmetic wrapper.
- `asr_align/export.py`: standalone F32 safetensors checkpoint export.
- `align_asr.py`: existing final-layer alignment fit/export CLI. Shared runs
  should use `--manifest`, not the legacy exploratory `--audio` path.
- `crosslingual_probe.py`: historical centered FLEURS retrieval probe. Shared
  runs should use `--manifest`, not the legacy `--fleurs` discovery path.
- `check_encoder_parity.py`: compares the PyTorch port with runtime debug
  embeddings; use it whenever graph, loading, precision, or export changes.
- `convert_asr_to_mmproj.py`: dependency-light safetensors/GGUF conversion
  utilities also reused by `asr_align/weights.py`.
- `dataset_a.py`: freezes Comparison 6's Gram audio. Extracts SLURP English
  assistant clips from the Zenodo tarball and Speech-MASSIVE fr/de/ru clips from
  the published parquet shards, resamples to 16 kHz, writes fixed-length crops,
  and freezes one manifest per corpus through `manifests.write_frozen`.
  `--common-voice` freezes the same budget from Common Voice instead, which is
  the off-domain Gram ablation. FLEURS and LibriSpeech appear in neither.
- `regmean_merge.py` and `asr_align/regmean.py`: the Comparison 6 runner and the
  closed-form merge. Tensor routing with an exactly-once assertion,
  frame-normalized Gram accumulation, Eq. 2 with the alpha shrinkage,
  RegMean++ Algorithm 1 in depth order, plain RegMean and simple averaging as
  reference arms, the LayerNorm-seeding ablation, held-out encoder-output
  agreement, and the delta table against comparisons 1 and 2. Nothing here is
  trained and `proj` is never touched. The evaluation passes are imported from
  the Comparison 2 runner so every arm's rows come from the same code.
- `lm_gating_check.py` and `asr_align/gating.py`: the blocking checks of
  `REGMEAN_INTERFACE_DESIGN.md` §11 — the encoder-width confirmation that sizes
  Comparison 6's Gram collection, the frozen MASSIVE sample, the two frozen
  output-language system prompts, both frozen-language-model teacher paths, the
  MASSIVE-fitted output-language identifier, the teacher-quality gate, and the
  text-path versus audio-path target gap.
- `tests/test_shared_setup.py`: unit coverage for shared invariants, manifests,
  and the evaluation contract.
- `tests/test_voice_assistant.py`: unit coverage for the speech-to-action
  manifest, candidate contract, scoring, and cross-candidate table.
- `tests/test_final_map.py`: unit coverage for the final activation maps, the
  projection fold, the frozen activation-cache reader, and the Comparison 3
  delta table.
- `tests/test_gating.py`: unit coverage for the encoder-width derivation, the
  frozen gating sample, the output-language identifier, the reply-shape rules,
  the per-cell rates, and the gate verdicts.
- `tests/test_regmean.py`: unit coverage for the tensor routing, the Gram
  normalization, Eq. 2 against a brute-force solve, the collapse onto the
  unweighted mean when the Grams coincide, the merge driver, the Dataset A
  manifests, and the extended result contract.

## Non-negotiable experiment invariants

1. Model arithmetic is only over shared canonical `encoder.*` tensors. Never
   include `proj.*`, featurizer tensors, decoder/joint tensors, or prompt
   projector tensors in a task vector.
2. Require identical tensor-key sets and exact shapes before arithmetic. Never
   rely on PyTorch or NumPy broadcasting. Reject missing, extra, NaN, or infinite
   tensors.
3. Perform arithmetic in F32 through `asr_align.experiments`. Load ASR sources
   with `mmproj_precision=False` and load FT_EN from the original NVIDIA
   safetensors. Quantize only a final exported artifact, then evaluate it again. When
   a projection is fitted through the frozen language model, the language-model
   precision used for fitting is itself an experimental variable: record it in
   the candidate's provenance and measure its effect separately from the export
   quantization stage.
4. Never use a dequantized Q8_0 VoiceChat container as the FT_EN arithmetic or
   reference source. `load_container()` remains for runtime-parity and legacy
   analysis only. Use `load_voicechat_safetensors()` for FT_EN; any F16/Q8_0
   rounding belongs to the final deployment conversion and must be measured as
   a separate post-quantization stage.
5. Every candidate uses an exact independent copy of the complete PT_ML runtime
   configuration. Do not inherit attention context or processor configuration
   from PT_EN or FT_EN. The known left contexts differ: PT_ML is 56 frames;
   PT_EN/FT_EN are 70.
6. Preserve the original VoiceChat/FT_EN projection unless a comparison
   explicitly fits a learned map and folds it in. Two forms are authorised: a
   closed-form reverse activation map fitted on embeddings, as in comparison 3;
   and a projection fitted by gradient descent through the frozen language
   model, as in comparison 7. A gradient-fitted projection must record its
   initialization, its frozen training manifest, the frozen system prompt it was
   fitted under, the language-model precision used for fitting, and the teacher
   path its targets were generated through, and must leave every `encoder.*`
   tensor byte-identical to its arm's source. Nothing
   else in the served graph may be trained. The multilingual checkpoint's
   language-prompt MLP is not part of the deployed VoiceChat graph.
7. Use the fixed lambda sweep from `asr_align.experiments`; do not add an ad hoc
   coefficient or select a final lambda before the specified development
   comparisons are complete.
8. Every candidate that reaches a deployment artifact also runs the paired
   speech-to-action evaluation. Retrieval never substitutes for it: a candidate
   whose foreign embeddings rank well may still leave the frozen language model
   unable to answer or call a tool.
9. Speech-to-action rows are comparable only under one frozen manifest, system
   prompt, response budget, runtime commit, runtime environment, and precision
   stage. `build_comparison()` enforces this; do not assemble a table by hand.
   The shared runtime environment file must not name `ASR_MODEL`, which differs
   per row and is verified through the server's discovery endpoint.

## Data and model-selection rules

- LibriSpeech is split by speaker into `map_train`, `validation`, and reserved
  `test`. Fit maps on `map_train`, select regularization on `validation`, and do
  not inspect the reserved test split while developing.
- FLEURS is evaluation-only. Never use FLEURS metrics, including final FLEURS
  results, to fit a map, choose regularization, choose lambda, or route tensors.
- A FLEURS pair must use distinct English reference and English query
  recordings. Never fall back to the same take twice.
- Shared runs consume manifests written by `shared_setup.py`. The consumers
  verify manifest digests plus each selected audio file's size and SHA-256.
- Frozen setup/manifests are write-once. If inputs or policies change, create a
  new experiment output directory instead of overwriting the old one.
- Generated models, embeddings, activations, datasets, and result caches belong
  under ignored paths such as `.cache/` or external model storage. Do not commit
  large artifacts.

## Evaluation contract

All comparisons must emit `asr_align.evaluation.evaluate_candidate` records
with the same frozen manifest hashes. Required sections are:

- English VoiceChat-space R2 and cosine against FT_EN;
- candidate-on-English retrieval;
- historical centered FLEURS retrieval;
- intrinsic candidate-to-candidate cross-lingual retrieval;
- top-1, top-5, MRR, median rank, hit count, N, and paired bootstrap confidence
  intervals versus the PT_ML baseline;
- embedding mean/norm diagnostics;
- an explicit `pre_quantization` or `post_quantization` stage.

Use `validate_precision_pair()` for artifacts requiring both precision stages.
Comparison 1 is the PT_ML reference; its paired deltas are zero. Later
comparisons must pass the exact frozen PT_ML predictions/embeddings as the
paired baseline, not rerun or reshuffle a separate baseline.

Every candidate exported to a deployment artifact additionally emits a
`asr_align.voice_assistant` result: exact single-call accuracy per language as
the primary endpoint, tool attempt/well-formedness/name/argument-type/value
breakdowns, required-fact scoring of the assistant text, English-output
compliance, and the paired Russian-minus-English difference. Each table also
carries the `FT_EN` control row, which bounds what the frozen language model can
do on this data and proves the harness elicits calls at all.

Retrieval is screening evidence. Do not make a deployment claim without the
actual ASR/VoiceChat evaluation required by the final checklist. The current
speech-to-action pilot is a development split: six single-call numeric cases
with synthesized Russian audio, and its results may be inspected before
prompt/model choices are made.

## Environment and reproducible setup

The host's default `python3` may not have NumPy, Torch, or SoundFile. Create the
locked repository-local project environment with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-align uv sync --python 3.12
```

`align_setup.sh` is retained for bootstrapping missing data and the external
runtime reader. It performs network access and force-cleans only that cached
runtime checkout; do not run it when a prepared reader and datasets already
exist. FLEURS and model artifacts are not guaranteed to be present.

Materialize a real shared setup before model comparisons:

```bash
cp shared_setup.example.json .cache/shared_setup.local.json
# Replace placeholder revisions, paths, and lineage evidence.
.venv-align/bin/python shared_setup.py \
  --spec .cache/shared_setup.local.json \
  --output .cache/experiments/shared-v1
```

The example file is intentionally invalid until all placeholder revisions and
evidence are replaced. Checkpoint paths alone are not provenance; keep the
pinned repo revision and SHA-256 artifact hashes in `shared_setup.json`. FT_EN
must point at the original NVIDIA safetensors checkpoint, never a Q8_0
container.

## Verification before committing

Serving a candidate for the speech-to-action evaluation uses the pinned
deployment runtime, one encoder per run:

```bash
ASR_MODEL=<candidate-name> docker compose \
  --env-file .cache/experiments/<pilot>/runtime.env \
  -f /tmp/nemotron-voicechat-main-<revision>/docker-compose.yml up -d voicechat
```

Wait for `backend_status: ready`, and let the runner check `asr_model` through
the server rather than assuming the restart took effect. Only one session is
allowed at a time, so runs are sequential.

For ordinary Python changes, run:

```bash
.venv-align/bin/python -m compileall -q asr_align *.py tests
.venv-align/bin/python -m unittest discover -s tests -v
git diff --check
```

For changes involving checkpoint loading, graph structure, precision, or
export, also run `check_encoder_parity.py` against a recorded runtime log. For
manifest changes, build twice into the same output and verify the second run is
idempotent; intentionally changed inputs must be rejected. For evaluator
changes, preserve the schema version or deliberately version and migrate it.

Before reporting an experiment complete, record the command, frozen setup and
manifest hashes, artifact hashes, precision stage, metrics, and reusable
embedding/activation outputs required by `EXPERIMENTS_TODO.md`.

## First steps for the next comparison

### Comparison 7 fitting-graph correction (2026-09-09)

The original `comparison-7-interface-v1` E1 fit used unit weight for the
function-token embedding in both the audio timeline and cached system prefix.
The original VoiceChat configuration (`model.stt.model`) and the pinned
runtime's function-head metadata instead require channel weights text=1,
audio=1, function=2. The runtime applies these weights even to PAD tokens.
The v1 English CE gate passed inside the mismatched graph; that verdict must
not authorize further fits. Preserve v1 artifacts and measurements as diagnostic
evidence rather than overwriting or promoting them.

The correction restores the existing served graph, with no invariant, data
budget, teacher, or training-objective change. Read all three weights from the
original checkpoint configuration, use the same F32 fusion operation for the
system prefix and audio timeline, and record the configuration hash and
`voicechat-duplex-fusion-v2` graph version in experiment/candidate provenance.
Start a fresh experiment at `comparison-7-interface-v2-fusion`, recalibrate its
loss weights from E1, refit E1 from its original initialization, and rerun the
English and deployment control checks before E2–E4. Frozen Dataset B, teacher
targets and encoder caches can be reused unchanged: none depends on the
student's channel-fusion operation. Do not resume a v1 optimizer checkpoint
under the corrected graph. The 25/50/100% budgets and two-epoch settings remain
unchanged. `COMPARISON_7_RESULTS.md` records the evidence and limits.
The specification's earlier claim that B1 hard-target CE is zero by
construction is also corrected: measure its initialization loss. This is a
clarification of the existing self-distillation objective, not a new loss.

### Comparison 7: the fusion fix is not why the pilot was silent (2026-09-09)

The correction above is kept, but the ablation it called for has since run and
shows it does not explain the v1 deployment failure: under the corrected graph
the v1 fitted projection still improves held-out English CE by about 0.090 nats
over the original in both prompt cells. **The corrected E1 refit is therefore
held**, because the supervision it would consume still carries the defect that
did cause the silence.

The measured cause is that this experiment straddles the runtime's two turn
paths. B1 targets were recorded through `vc_session::run_turn`, the whole-wav
path, which honours `VC_FORCE_BOS` and passes `a = nullptr` after the wav — an
exact zero audio embedding, a convention `interface_fit.duplex_inputs`
reproduces. The deployment pilot runs `vc_session::duplex_step` through the
Realtime bridge, which clears `hold_bos` and `want_bos` every frame and feeds
encoded PCM silence on an input underrun. Given the tail deployment actually
sends, the v1 fitted projection never opens a turn on 24/24 held-out English
turns; the original answers 24/24.

Two further defects follow from the same place and also need a decision before
any arm is refit: the objective is a uniform mean over frames that are 63.7%
PAD, so 64% of the v1 fit's measured gain sits on two per-trace onset frames —
a constant EOS at frame 9 and the forced BOS — 39% on more confident silence,
and the reply content is slightly worse; and the English gate is that same
uniform mean, so it certified −0.11 nats for a projection that cannot open a
turn. Do not treat a gate pass as a deployment result until the gate includes a
free-running check on the duplex path with an encoded-silence tail.
`COMPARISON_7_RESULTS.md` records the four-cell measurement and the artifacts.

`simple-average`'s exported artifact in the Comparison 6 output is arm `E4`'s
encoder for Comparison 7 and is already evaluated pre-quantization; reuse it
rather than rebuilding it. Comparison 6 also freezes Dataset A, whose SLURP
`train` and Speech-MASSIVE `dev` draws Comparison 7's Dataset B must stay
disjoint from: the utterance ids are in the frozen manifests under
`.cache/experiments/dataset-a-v1/`.

Comparison 4 starts from the same frozen setup and Comparison 1 references.
Reuse Comparison 3's selected final reverse map in the projection and verify
the lambda=0 candidate against Comparison 3. Follow the dense-transport
checklist for collecting internal representations, fitting maps on `map_train`,
selecting regularization on `validation`, and measuring structured-update
residuals plus held-out layerwise transported-update agreement. FLEURS remains
evaluation-only and must not select maps, regularization, lambda, or routing.

Every arm with a learned projection must export a directory containing its
folded `proj.*` and featurizer tensors. Without them
the deployment converter falls back to the container's own projection and the
served artifact silently stops being the candidate. After exporting, serve the
Q8 artifact and add its row to the speech-to-action table alongside the existing
`FT_EN` control and comparison 1--3 rows. Create a new table output directory;
the existing frozen tables must remain unchanged.
