"""The deployment's mel featurizer, ported from C++.

`mtmd.cpp` gives PROJECTOR_TYPE_VOICECHAT the parakeet preprocessor with
`norm_per_feature = false`:

    audio_preproc = std::make_unique<mtmd_audio_preprocessor_parakeet>(ctx_a, false);

so the features the encoder actually sees are the ones
`mtmd_audio_preprocessor_parakeet::preprocess` computes, minus the per-feature
normalization at the end. This module reproduces that function exactly. It has
to be exact rather than merely equivalent-in-spirit, because the whole point of
the alignment is to measure what one encoder does to *these* inputs against what
the other does to them; a featurizer that differed even slightly would make the
fitted map correct for an input distribution that is not the deployed one.

Two consequences worth stating, because they are easy to miss:

  * **No mean/variance normalization.** NeMo's default preprocessor normalizes
    each mel bin over the utterance, and both published ASR checkpoints were
    trained that way. VoiceChat's config says `normalize: "NA"`, so the graph
    does not, and the swapped-in encoder is therefore fed a distribution it was
    never trained on. `per_feature_normalize()` here exists to measure how much
    that costs: it is *not* what the deployment does, and turning it on in
    production would be a one-argument change in `mtmd.cpp` plus a new mmproj
    key. See README, "Interface alignment".

  * **The filterbank and the window come from the container**, not from a
    formula. `mtmd_audio_preprocessor_parakeet::initialize` copies
    `hparams.mel_filters` and `hparams.window` straight out of the mmproj, which
    the converters copy out of `stt_model.perception.preprocessor.featurizer`.
"""

from __future__ import annotations

import numpy as np
import torch

# mtmd_audio_preprocessor_parakeet::preprocess, filter_params
SAMPLE_RATE = 16000
N_FFT = 512
WIN_LENGTH = 400
HOP_LENGTH = 160
PREEMPH = 0.97
# const double eps = 5.960464477539063e-08 in worker_thread; 2^-24.
LOG_EPS = 5.960464477539063e-08


def preemphasis(samples: torch.Tensor) -> torch.Tensor:
    """x[i] -= 0.97 * x[i-1], leaving x[0] alone.

    The C++ runs the loop backwards in place, so every read of ``x[i-1]`` sees
    the original sample; that is the same thing as this one shifted difference.
    """

    out = samples.clone()
    out[..., 1:] = samples[..., 1:] - PREEMPH * samples[..., :-1]
    return out


def log_mel(
    samples: torch.Tensor,
    mel_filters: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    """One waveform to ``(n_mel, n_frames)`` log-mel, the way the graph sees it.

    ``samples`` is 1-D and already at 16 kHz. ``mel_filters`` is ``(n_mel,
    n_fft // 2 + 1)`` and ``window`` is ``(400,)``, both out of the container.
    """

    if samples.ndim != 1:
        raise ValueError(f"log_mel takes one waveform, got shape {tuple(samples.shape)}")
    dtype = torch.float64
    samples = preemphasis(samples.to(dtype))

    # "Parakeet uses centered constant padding": frame_size / 2 zeros each side.
    pad = N_FFT // 2
    padded = torch.nn.functional.pad(samples, (pad, pad))

    n_frames = (padded.shape[0] - N_FFT) // HOP_LENGTH + 1
    if n_frames <= 0:
        raise ValueError(f"audio of {samples.shape[0]} samples is shorter than one frame")

    # Each frame is written into a zeroed 512-point buffer starting at
    # window_pad_left, and the samples are read from the same offset -- so the
    # windowed 400 samples sit centred in the FFT input and the frame's centre
    # lands on padded[i * hop + 256], i.e. on the original sample i * hop.
    window_pad_left = (N_FFT - WIN_LENGTH) // 2
    starts = torch.arange(n_frames) * HOP_LENGTH + window_pad_left
    idx = starts[:, None] + torch.arange(WIN_LENGTH)[None, :]
    frames = padded[idx] * window.to(dtype)

    buffer = torch.zeros(n_frames, N_FFT, dtype=dtype)
    buffer[:, window_pad_left:window_pad_left + WIN_LENGTH] = frames

    # modulus^2 of the real FFT, bins 0 .. n_fft / 2
    spectrum = torch.fft.rfft(buffer, n=N_FFT, dim=-1)
    power = spectrum.real.square() + spectrum.imag.square()

    mel = power @ mel_filters.to(dtype).t()
    return torch.log(mel + LOG_EPS).t().contiguous()


def per_feature_normalize(mel: torch.Tensor) -> torch.Tensor:
    """NeMo's `normalize: per_feature`, which the VoiceChat graph does NOT do.

    Kept because it is the single cheapest hypothesis for why a published ASR
    encoder goes deaf in this pipeline, and because measuring it costs one flag.
    The C++ divides by ``std + 1e-5`` with an n-1 variance over the valid frames
    only; there is no padding here, so every frame is valid.
    """

    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, unbiased=True, keepdim=True)
    return (mel - mean) / (std + 1e-5)


def frames_out(n_samples: int) -> int:
    """Encoder frames a waveform of ``n_samples`` produces, at 12.5 Hz."""

    n_mel_frames = (n_samples + 2 * (N_FFT // 2) - N_FFT) // HOP_LENGTH + 1
    # three stride-2 convolutions, each padding 2 left and 1 right of a 3-wide
    # kernel: out = (in + 2 + 1 - 3) // 2 + 1 = in // 2 + 1
    for _ in range(3):
        n_mel_frames = n_mel_frames // 2 + 1
    return n_mel_frames


def as_numpy(mel: torch.Tensor) -> np.ndarray:
    return mel.detach().cpu().numpy()
