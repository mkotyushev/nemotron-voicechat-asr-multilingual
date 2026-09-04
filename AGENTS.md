# Repository guide for coding agents

Read this file, `EXPERIMENTS_TODO.md`, and the relevant CLI/module before
changing code. `EXPERIMENTS_TODO.md` is the current experiment specification;
complete and check off one comparison at a time. A checked implementation item
does not imply that a model experiment has been run.

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

The Shared setup implementation is complete. Comparisons 1--5 and the final
comparison remain experimental work; do not mark their result-oriented boxes
complete without producing and validating the stated artifacts and metrics.

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
- `asr_align/weights.py`: maps ASR safetensors and the VoiceChat GGUF onto one
  canonical state-dict naming scheme.
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
- `tests/test_shared_setup.py`: unit coverage for shared invariants, manifests,
  and the evaluation contract.

## Non-negotiable experiment invariants

1. Model arithmetic is only over shared canonical `encoder.*` tensors. Never
   include `proj.*`, featurizer tensors, decoder/joint tensors, or prompt
   projector tensors in a task vector.
2. Require identical tensor-key sets and exact shapes before arithmetic. Never
   rely on PyTorch or NumPy broadcasting. Reject missing, extra, NaN, or infinite
   tensors.
3. Perform arithmetic in F32 through `asr_align.experiments`. Load ASR sources
   with `mmproj_precision=False` for arithmetic. Quantize only a final exported
   artifact, then evaluate it again.
4. `load_asr()` and `load_container()` default to `mmproj_precision=True` for
   runtime-parity work. That deliberately simulates the converter's F16/Q8_0
   rounding. Do not accidentally use that default for task-vector construction.
   The FT_EN GGUF is already quantized at source; dequantizing it to F32 cannot
   recover an unavailable pre-quantization checkpoint, so record that fact.
5. Every candidate uses an exact independent copy of the complete PT_ML runtime
   configuration. Do not inherit attention context or processor configuration
   from PT_EN or FT_EN. The known left contexts differ: PT_ML is 56 frames;
   PT_EN/FT_EN are 70.
6. Preserve the original VoiceChat/FT_EN projection unless a comparison
   explicitly folds a learned reverse activation map into it. The multilingual
   checkpoint's language-prompt MLP is not part of the deployed VoiceChat graph.
7. Use the fixed lambda sweep from `asr_align.experiments`; do not add an ad hoc
   coefficient or select a final lambda before the specified development
   comparisons are complete.

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

Retrieval is screening evidence. Do not make a deployment claim without the
actual ASR/VoiceChat evaluation required by the final checklist.

## Environment and reproducible setup

The host's default `python3` may not have NumPy, Torch, or SoundFile. Prefer the
project environment created by:

```bash
cp .env.example .env
./align_setup.sh
```

`align_setup.sh` creates `.venv-align`, installs CUDA Torch and dependencies,
pins/patches an external runtime reader under `.cache`, and downloads
LibriSpeech. It performs network access and force-cleans only that cached
runtime checkout; do not run it casually when a prepared environment already
exists. FLEURS and model artifacts are not guaranteed to be present.

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
pinned repo revision and SHA-256 artifact hashes in `shared_setup.json`.

## Verification before committing

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

For comparison 1, start from the frozen shared setup rather than adding another
configuration system. Load the unmodified PT_ML encoder, attach the original
FT_EN VoiceChat projection, run a deterministic finite-output check, export and
reload a pass-through copy, verify pre-quantization encoder equality, quantify
deployment quantization, run the entire common evaluator, and save pooled plus
per-layer activations for reuse. Do not refit an interface map in this baseline.
