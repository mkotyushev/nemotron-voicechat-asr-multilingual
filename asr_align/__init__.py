"""Align a standalone NVIDIA streaming ASR encoder with VoiceChat's own.

`convert_asr_to_mmproj.py` can already build the perception mmproj out of
`nvidia/nemotron-3.5-asr-streaming-0.6b` instead of the VoiceChat container.
The result loads and runs and produces nothing the language model can read,
because the 1024 -> 4480 `proj` it keeps from the container was trained against
the container's own encoder and the multilingual encoder's output space has
drifted away from it (cosine 0.63-0.74 in the deep layers, against 0.86-0.93
for the English encoder VoiceChat was fine-tuned from, which does work).

This package measures that drift on real speech and undoes as much of it as a
linear map can, following Singh and Jaggi, "Model Fusion via Optimal Transport"
(NeurIPS 2020) and the implementation of it in `chmv2_distill/ot_width.py`.

    features.py    the deployment's mel featurizer, ported exactly
    weights.py     one naming for two weight sources: safetensors and container
    encoder.py     the deployed graph as a PyTorch module, with hook points
    data.py        English calibration speech
    transport.py   the optimal-transport core, adapted from ot_width.py
    interface.py   the encoder -> LLM map, which is what actually gets shipped
    fuse.py        OT-aligned weight interpolation between the two encoders

Read `align_asr.py` for how they fit together and README, "Interface
alignment", for why each piece is shaped the way it is.
"""
