# Multilingual ASR experiments for NemotronLabs VoiceChat

Experimental work on replacing the perception encoder inside
[`NVIDIA-NemotronLabs-VoiceChat-11B`](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B)
with NVIDIA's multilingual streaming ASR encoder.

The serving integration lives in
[`nemotron-voicechat-realtime-gguf`](https://github.com/mkotyushev/nemotron-voicechat-realtime-gguf).
This repository contains the unfinished alignment research, measurements, and
checkpoint export code.

## Status

This is not a finished multilingual VoiceChat model and is not the recommended
server default.

- The published multilingual encoder loads, but VoiceChat does not understand
  its output space without an additional map.
- A fitted linear map restored one English test answer and improved held-out
  English embedding agreement.
- The map reduced some of the multilingual information measured by retrieval.
- The orthogonal map performed worse than ridge regression.
- Direct encoder-only task arithmetic has been measured end to end. Small
  coefficients trade modest English-space gains against retention, while the
  larger coefficients sharply reduce multilingual retrieval.
- Measured through the deployed server on spoken tool-calling requests, neither
  the unmodified multilingual encoder nor direct task arithmetic at the primary
  coefficient produces any assistant turn at all, in either language. The
  original encoder answers on the same clips and calls the right tool on four of
  six English cases, so the harness and prompt are not the obstacle. That same
  control answers all six Russian clips in English while never attempting a
  tool call, having plainly misheard them.

The useful result so far is a reproducible experiment and an ordinary exported
checkpoint. It is not evidence that the model is generally multilingual, and the
end-to-end measurement is currently a negative result: the encoder-space metrics
that improve are not yet enough for the language model to respond.

## Layout

| Path | Purpose |
|---|---|
| `align_asr.py` | fit, evaluate, and export an interface map |
| `shared_setup.py` | pin checkpoints, validate arithmetic, and freeze all shared manifests |
| `pt_ml_baseline.py` | run Comparison 1 and freeze the PT_ML reference artifacts |
| `asr_align/baseline.py` | PT_ML pass-through, provenance, equality, and precision checks |
| `direct_task_arithmetic.py` | run the fixed Comparison 2 task-arithmetic sweep |
| `asr_align/direct.py` | task-vector reports, baseline validation, growth checks, and Pareto tables |
| `asr_align/experiments.py` | checkpoint roles, strict F32 encoder arithmetic, and provenance |
| `asr_align/manifests.py` | immutable speaker/sentence/take manifests |
| `asr_align/evaluation.py` | common comparison metrics, diagnostics, and paired intervals |
| `voice_assistant_evaluation.py` | freeze the paired speech pilot and score a served candidate |
| `asr_align/voice_assistant.py` | pre-TTS tool-call scoring, candidate contract, and the cross-candidate table |
| `asr_align/encoder.py` | PyTorch port of the runtime FastConformer graph |
| `asr_align/features.py` | matching audio featurizer |
| `asr_align/interface.py` | interface-map fits and held-out scoring |
| `asr_align/transport.py` | activation-based transport/permutation plans |
| `asr_align/fuse.py` | unfinished weight-space fusion arm |
| `asr_align/export.py` | write a standalone aligned or pass-through checkpoint |
| `check_encoder_parity.py` | compare the PyTorch port with runtime embeddings |
| `crosslingual_probe.py` | multilingual retrieval probe |
| `convert_asr_to_mmproj.py` | inspect or convert an exported checkpoint |
| `align_setup.sh` | environment, runtime reader, and LibriSpeech setup |

## Inputs

The experiment needs three model artifacts:

1. `E = PT_EN`: `nvidia/nemotron-speech-streaming-en-0.6b`.
2. `M = PT_ML`: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
3. `F = FT_EN`: the original NVIDIA VoiceChat `model.safetensors` checkpoint.

Do not use a Q8_0 container dequantized back to F32 as `FT_EN`; the lost
precision cannot be recovered. GGUF quantization is only applied to a final
export immediately before deployment evaluation.

The server conversion additionally needs its tokenizer files.

The sibling server repo can download them. In its `.env`, temporarily set:

```bash
ASR_MODEL=multilingual
```

Then run its downloader:

```bash
./download.sh
```

The default `.env.example` here points `MODEL_DIR` at that sibling repo's
`models/` directory. Change it if the artifacts live elsewhere.

## Setup

```bash
uv init --bare --name nemotron-voicechat-asr-multilingual --python 3.12
UV_PROJECT_ENVIRONMENT=.venv-align uv sync --python 3.12
```

The locked project environment contains CUDA PyTorch and the scientific/audio
dependencies. `align_setup.sh` remains available to download LibriSpeech and
prepare a runtime reader when those assets are not already present.

## Frozen shared experiment setup

Before running comparisons, copy the shared spec into the ignored cache area,
replace every checkpoint revision with an immutable revision, cite the source
that identifies `PT_EN` as the intended `FT_EN` ancestor, and point it at a
local FLEURS tree:

```bash
cp shared_setup.example.json .cache/shared_setup.local.json
# edit .cache/shared_setup.local.json
.venv-align/bin/python shared_setup.py \
    --spec .cache/shared_setup.local.json \
    --output .cache/experiments/shared-v1
```

This command fails closed on a movable or missing revision, wrong model role,
lineage mismatch, missing/non-finite/broadcastable tensors, changed runtime
configuration, speaker leakage, reused English FLEURS takes, or an attempt to
replace an already-frozen setup. It writes:

```text
.cache/experiments/shared-v1/
  shared_setup.json          checkpoint revisions, hashes, configs and policies
  manifests/librispeech.json
  manifests/fleurs.json
```

The LibriSpeech manifest has speaker-disjoint `map_train`, `validation`, and
reserved `test` splits. FLEURS is evaluation-only. Every audio record is pinned
by path, byte count, and SHA-256; consumers verify those values before a run.
All model arithmetic is over exact canonical `encoder.*` tensors in F32, using
the fixed lambda sweep `0, .25, .5, .75, 1` with `1` as the primary endpoint.
FT_EN is read from the original VoiceChat safetensors; only final exported
artifacts may be quantized.

Run the fit with:

```bash
set -a; source .env; set +a
.venv-align/bin/python align_asr.py \
    --asr-dir   "$MODEL_DIR/asr-multilingual" \
    --container "$MODEL_DIR/nemotron_voicechat_11b-Q8_0.gguf" \
    --manifest  .cache/experiments/shared-v1/manifests/librispeech.json \
    --work      "$CONVERT_WORK" \
    -o          "$MODEL_DIR/asr-multilingual-aligned"
```

The output directory is a standalone checkpoint. It includes the multilingual
encoder, the fitted VoiceChat projection, and the mel-featurizer tensors needed
by the server converter.

All comparisons emit records through `asr_align.evaluation.evaluate_candidate`.
That contract always includes English VoiceChat-space R²/cosine, the three
retrieval views, top-1/top-5/MRR/median rank/hit count/N, paired bootstrap
intervals against `PT_ML`, embedding mean/norm diagnostics, manifest hashes,
and an explicit pre- or post-quantization stage.

### Comparison 1: PT_ML baseline

Run the unmodified multilingual reference only after `shared_setup.py` has
materialized a real frozen setup:

```bash
.venv-align/bin/python pt_ml_baseline.py \
    --shared-setup .cache/experiments/shared-v1/shared_setup.json \
    --work .cache/llama-voicechat.cpp \
    --device cuda \
    -o .cache/experiments/comparison-1-pt-ml
```

The runner refuses to replace an existing output. It rehashes the pinned
checkpoints and manifests, uses the exact `PT_ML` runtime configuration, and
never fits an alignment map. It exports a self-contained F32 pass-through
checkpoint, converts it to a real Q8_0 perception GGUF, reloads the GGUF, and
requires its tensors to exactly match the converter-rounding model before
running both precision stages through the common evaluator.

The output records the command and hashes in `run.json`, writes pre/post result
records under `results/`, saves the exact arrays later comparisons must use as
their paired PT_ML reference under `embeddings/`, and shards the pre-quantized
subsampling/block outputs for LibriSpeech `map_train` and `validation` under
`activations/`. The reserved LibriSpeech `test` split is not encoded. Optional
`--parity-wav` plus `--runtime-log` runs the recorded runtime parity check as
part of the experiment.

### Comparison 2: direct task arithmetic

Run the complete fixed sweep only after Comparison 1 has produced its frozen
paired-reference arrays:

```bash
.venv-align/bin/python direct_task_arithmetic.py \
    --shared-setup .cache/experiments/shared-v1/shared_setup.json \
    --baseline .cache/experiments/comparison-1-pt-ml \
    --work .cache/llama-voicechat.cpp \
    --device cuda \
    -o .cache/experiments/comparison-2-direct
```

The runner computes `FT_EN - PT_EN` only over canonical `encoder.*` tensors in
F32, records norms by block/module/tensor, verifies reconstruction, exports and
evaluates all five predefined lambdas, and checks lambda zero exactly against
Comparison 1. The original VoiceChat projection and complete PT_ML runtime
configuration remain unchanged. Only the primary lambda-one artifact is
converted to Q8_0 and reevaluated. The development Pareto table is recorded
without selecting a final lambda.

### Paired English/Russian speech-to-action tool calling

Retrieval and VoiceChat-space agreement are measured inside the encoder. They
cannot say whether the frozen language model still understands the request and
emits the right tool call, which is what the alignment exists to achieve. Every
candidate that reaches a deployment artifact therefore also runs through the
pinned deployment server and is scored at the boundary before TTS: the decoded
assistant text and the parsed structured call. Generated speech is discarded.

Freeze the dataset once. English uses the official AU-Harness BFCL-v3
recordings; Russian uses pinned Silero synthesis of the RFCB translations of the
same cases, so the expected call is identical in both languages:

```bash
.venv-align/bin/python voice_assistant_evaluation.py prepare \
    --rfcb .cache/datasets/RFCB \
    --silero .cache/tools/silero-models \
    --bfcl-audio .cache/datasets/BFCL_v3_audio \
    --bfcl-audio-revision <immutable-dataset-commit> \
    --russian-model <silero-ru-model.pt> \
    --output .cache/experiments/voice-assistant-pilot-v2/dataset
```

Serve one encoder at a time from the pinned runtime, then score it. A run pins
the served artifact, the runtime commit, and the runtime environment file, and
verifies through the server which encoder is actually loaded:

```bash
ASR_MODEL=direct-lambda-1 docker compose \
    --env-file .cache/experiments/voice-assistant-pilot-v2/runtime.env \
    -f /tmp/nemotron-voicechat-main-<revision>/docker-compose.yml up -d voicechat

.venv-align/bin/python voice_assistant_evaluation.py run \
    --manifest .cache/experiments/voice-assistant-pilot-v2/dataset/manifest.json \
    --expected-asr-model direct-lambda-1 \
    --candidate-id direct-lambda-1 --comparison 2 \
    --precision post_quantization \
    --artifact <served-mmproj.gguf> \
    --shared-setup .cache/experiments/comparison-2-direct-v1/shared_setup.json \
    --runtime-repository ~/model-deployments/nemotron-voicechat \
    --runtime-revision <immutable-runtime-commit> \
    --runtime-env .cache/experiments/voice-assistant-pilot-v2/runtime.env \
    --output .cache/experiments/voice-assistant-pilot-v2/direct-lambda-1-post-q8
```

The primary endpoint is exact single-call accuracy per language. Tool attempt,
well-formedness, tool name, argument types, and argument values are recorded
separately, as are required-fact matches in the assistant text and English-output
compliance. Each language is scored against the canonical expected call rather
than against the other language, so an identical failure in both cannot look
like cross-lingual success.

The scored text is what the model hands to TTS, so it spells numbers out. Fact
matching therefore reads spoken English numerals -- "one hundred and twenty"
satisfies a required `120` -- while a bare numeric fact still has to appear as
its own token, so `10` is not evidence for `1`. Rounding is not forgiven: a
response saying `8.85` does not satisfy a required `8.854`. A call naming the
right tool with the wrong separator is scored wrong, because it is not
dispatchable, but it is flagged as `tool_name_separator_variant` so a near miss
is distinguishable from calling a different tool.

`compare` collects the scored rows into one table and refuses to combine rows
produced under different manifests, prompts, response budgets, runtimes, or
precision stages. Every table also carries an `FT_EN` control row served by the
original VoiceChat encoder, which bounds what the frozen language model can do
on this data.

The response budget is measured from the end of the audio rather than from the
start of the session. Measuring it from the session start would give a long clip
less decoding time than a short one, and the Russian clips are systematically
shorter than the English ones -- a bias directly on the cross-language quantity
being scored.

This is a development pilot: six single-call numeric cases with single-speaker
synthesized Russian audio. It detects gross differences, and it may not be used
to select a candidate for a deployment claim.

## Why the encoder cannot simply be swapped

VoiceChat's perception encoder is a causal FastConformer. NVIDIA publishes
closely related English and multilingual streaming ASR encoders with the same
main graph shape: 24 layers, width 1024, eight heads, FFN width 4096, 128 mel
bins, and subsampling factor eight.

Matching shapes are not matching representations. VoiceChat's `proj` was
trained to map its own encoder's 1024-wide output into the language model's
4480-wide embedding space.

The standalone English encoder remains close enough to work in the tested
question. The multilingual encoder has moved far enough that the same projection
behaves as if useful speech was not heard.

Cosine similarity between selected weights illustrates the drift:

| Tensor | VC~EN | VC~ML | EN~ML |
|---|---:|---:|---:|
| first-block Q projection | 0.855 | 0.742 | 0.737 |
| block-12 V projection | 0.909 | 0.717 | 0.686 |
| block-23 FFN input | 0.934 | 0.630 | 0.621 |
| subsampling linear | 0.966 | 0.930 | 0.944 |

The multilingual model also has a language-prompt MLP after the encoder. The
VoiceChat graph has no corresponding input, so the current swap omits it.

## Interface alignment

The encoder's final operation is a normalization followed by VoiceChat's linear
projection:

```text
embedding = hidden @ proj.weight.T + proj.bias
```

A fitted affine map `(M, m)` can therefore be folded into `proj` without a
runtime graph change:

```text
proj.weight <- proj.weight @ M
proj.bias   <- original_proj.weight @ m + proj.bias
```

`align_asr.py` records both encoders on the same English speech, holds out whole
speakers, fits several maps, and exports the selected result as normal
safetensors.

Inside the encoder, arbitrary basis changes do not survive the LayerNorms.
Transport plans are therefore used only to test whether neurons were permuted
and to support the separate fusion experiment.

## Port parity

The PyTorch encoder is intended to match the runtime graph, including Q8_0
rounding and the converter's F16 convolution kernels. Verify that before
trusting a fit:

```bash
.venv-align/bin/python check_encoder_parity.py \
    --asr-dir   "$MODEL_DIR/asr-multilingual" \
    --container "$MODEL_DIR/nemotron_voicechat_11b-Q8_0.gguf" \
    --work      "$CONVERT_WORK" \
    --audio     /path/to/test.wav
```

Measured agreement for the same 7.04 s clip:

| Comparison | MAE | Max error | Cosine |
|---|---:|---:|---:|
| container port vs runtime | 0.000182 | 0.00324 | 0.99999 |
| standalone encoder port vs runtime | 0.000183 | 0.00287 | 0.99999 |

## Alignment results

The transport plans were effectively the identity. Continued training moved
weights but did not meaningfully renumber neurons.

| Residual stream | Transport cost | Identity cost | Same index |
|---|---:|---:|---:|
| English vs container | 0.2259 | 0.2263 | 1.00 |
| Multilingual vs container | 0.3379 | 0.3387 | 1.00 |

Held-out embedding-space scores:

| Map | Multilingual R² | Cosine | English R² | Cosine |
|---|---:|---:|---:|---:|
| identity | -0.708 | 0.298 | +0.037 | 0.593 |
| permutation | -0.698 | 0.300 | +0.031 | 0.590 |
| orthogonal | -0.005 | 0.477 | -0.005 | 0.613 |
| dense ridge | +0.272 | 0.564 | +0.326 | 0.746 |
| affine ridge | +0.279 | 0.568 | +0.331 | 0.749 |

The projection-quantized score for the selected fit moved only slightly, which
suggests Q8_0 projection rounding is not the main limitation.

On the single spoken question used as the serving check:

| Perception file | Reply |
|---|---|
| container | correct answer |
| raw multilingual | unrelated answer |
| multilingual plus fitted map | correct answer |

That is a narrow sanity check, not a multilingual evaluation.

## Cross-lingual probe

Because the map is fitted on English, `crosslingual_probe.py` asks whether
foreign speech still retrieves its matching English translation in the
4480-wide embedding space.

Measured top-1 retrieval:

| Encoder/interface | French | German | Russian |
|---|---:|---:|---:|
| English encoder ceiling | 99.3% | 99.3% | 99.3% |
| VoiceChat container floor | 19.7% | 9.3% | 19.0% |
| raw multilingual | 57.7% | 35.3% | 42.9% |
| multilingual plus fitted map | 42.3% | 32.7% | 36.1% |
| chance | 0.7% | 0.7% | 0.7% |

The multilingual encoder carries useful cross-lingual structure, but the map
loses some of it while moving the representation into the space VoiceChat can
read. Live non-English speech quality remains unresolved.

## Weight-space fusion

`asr_align/fuse.py` explores a second route:

```text
fused = VoiceChat + lambda * (multilingual - English)
```

The transport plans justify tensor correspondence, and the interface map can be
refitted on top of a fused encoder. This arm has not been measured end to end.

Do not treat the fusion code or its default coefficients as a usable model.

## Serving an exported checkpoint

In the server repo's `.env`:

```bash
ASR_MODEL=multilingual-aligned
ASR_DIR=/path/to/asr-multilingual-aligned
```

Then rebuild the perception GGUF and recreate the container:

```bash
./convert.sh --force
docker compose up -d
```

The server repo intentionally knows only how to consume the checkpoint. The
fit, probes, fusion work, and research claims stay here.

## License

The code in this repository is MIT-licensed. Model weights are not included and
retain their upstream licenses and notices; see [LICENSE](LICENSE) and
[NOTICE](NOTICE).
