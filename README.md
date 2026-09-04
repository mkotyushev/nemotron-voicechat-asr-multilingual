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
- Weight-space fusion has not been measured end to end.

The useful result so far is a reproducible experiment and an ordinary exported
checkpoint. It is not evidence that the model is generally multilingual.

## Layout

| Path | Purpose |
|---|---|
| `align_asr.py` | fit, evaluate, and export an interface map |
| `shared_setup.py` | pin checkpoints, validate arithmetic, and freeze all shared manifests |
| `asr_align/experiments.py` | checkpoint roles, strict F32 encoder arithmetic, and provenance |
| `asr_align/manifests.py` | immutable speaker/sentence/take manifests |
| `asr_align/evaluation.py` | common comparison metrics, diagnostics, and paired intervals |
| `asr_align/encoder.py` | PyTorch port of the runtime FastConformer graph |
| `asr_align/features.py` | matching audio featurizer |
| `asr_align/interface.py` | interface-map fits and held-out scoring |
| `asr_align/transport.py` | activation-based transport/permutation plans |
| `asr_align/fuse.py` | unfinished weight-space fusion arm |
| `asr_align/export.py` | write a standalone aligned checkpoint |
| `check_encoder_parity.py` | compare the PyTorch port with runtime embeddings |
| `crosslingual_probe.py` | multilingual retrieval probe |
| `convert_asr_to_mmproj.py` | inspect or convert an exported checkpoint |
| `align_setup.sh` | environment, runtime reader, and LibriSpeech setup |

## Inputs

The experiment needs three model artifacts:

1. `E = PT_EN`: `nvidia/nemotron-speech-streaming-en-0.6b`.
2. `M = PT_ML`: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
3. `F = FT_EN`: the VoiceChat source container.

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
cp .env.example .env
./align_setup.sh
```

The setup script creates a separate CUDA PyTorch environment, downloads
LibriSpeech `dev-clean`, and prepares the pinned runtime reader used for parity
checks. It does not modify the server environment.

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
Only final exported artifacts may be quantized.

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
