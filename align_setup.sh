#!/usr/bin/env bash
# Prepare the experimental alignment environment, calibration speech, and the
# pinned llama-voicechat.cpp readers used for parity checks.
set -euo pipefail

cd "$(dirname "$0")"
# shellcheck disable=SC1091
source .env

HERE=$(pwd)
VENV=${ALIGN_VENV:-$HERE/.venv-align}
DATA=${ALIGN_DATA:-$HERE/.cache/data}
WORK=${CONVERT_WORK:-$HERE/.cache/llama-voicechat.cpp}
UV=${UV:-uv}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}

command -v "$UV" >/dev/null 2>&1 || { echo "no uv at $UV — set UV=" >&2; exit 1; }

# ----------------------------------------------------------------- the venv
if [ ! -x "$VENV/bin/python" ]; then
    echo "== creating $VENV =="
    # 3.12 to match the deployment's .venv, so anything learned about one
    # environment transfers to the other.
    "$UV" venv --python 3.12 "$VENV"
fi

if ! "$VENV/bin/python" -c "import torch" 2>/dev/null; then
    echo "== installing torch from $TORCH_INDEX =="
    VIRTUAL_ENV="$VENV" "$UV" pip install --index-url "$TORCH_INDEX" torch
fi

echo "== installing the rest =="
# numpy and scipy for the transport solvers, POT for the exact barycentric plan
# (--column-assignment barycentric only; the integral one is scipy's), and
# soundfile to read LibriSpeech's flac.
VIRTUAL_ENV="$VENV" "$UV" pip install numpy scipy pot soundfile

"$VENV/bin/python" - <<'PY'
import torch
print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
      f"available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}")
PY

# ------------------------------------------------------------ runtime reader
if [ ! -d "$WORK/.git" ]; then
    echo "== cloning llama-voicechat.cpp =="
    mkdir -p "$(dirname "$WORK")"
    git clone --filter=blob:none -b voicechat \
        https://github.com/sansamour/llama-voicechat.cpp.git "$WORK"
fi

git -C "$WORK" checkout --quiet --force "$VC_REF"
git -C "$WORK" clean -qfd tools/voicechat
git -C "$WORK" apply "$HERE/patches/q8_0-converters.patch"

# ------------------------------------------------------- calibration speech
# LibriSpeech dev-clean: 337 MB, 5.4 hours, 40 speakers, already 16 kHz. Read
# speech rather than conversation, which is closer to how someone talks to this
# deployment than a telephone corpus would be. align_asr.py holds whole speakers
# out of the fit, so the 40 matter more than the hours.
SPEECH="$DATA/LibriSpeech/dev-clean"
if [ ! -d "$SPEECH" ]; then
    echo "== fetching LibriSpeech dev-clean into $DATA =="
    mkdir -p "$DATA"
    curl -L --fail -o "$DATA/dev-clean.tar.gz" https://www.openslr.org/resources/12/dev-clean.tar.gz
    tar -xzf "$DATA/dev-clean.tar.gz" -C "$DATA"
    rm -f "$DATA/dev-clean.tar.gz"
fi
echo "speech: $SPEECH ($(find "$SPEECH" -name '*.flac' | wc -l) files)"

cat <<EOF

== done ==
Next, with the server stopped or with room on the GPU:

  $VENV/bin/python align_asr.py \\
      --asr-dir   ${MODEL_DIR:-$HERE/models}/asr-multilingual \\
      --container ${MODEL_DIR:-$HERE/models}/nemotron_voicechat_11b-${SRC_QUANT:-Q8_0}.gguf \\
      --audio     $SPEECH \\
      --work      $WORK \\
      -o          ${MODEL_DIR:-$HERE/models}/asr-multilingual-aligned

That writes a checkpoint directory the server converter can consume.
EOF
