#!/usr/bin/env bash
# One-time environment setup. Needs the NVIDIA driver on the Windows host;
# WSL picks up CUDA through it, no driver install inside the distro.
#
# This is the only shell script left, and it is shell because it is the one
# thing that cannot be Python: it creates the interpreter everything else runs
# in. Anything that runs *after* .venv exists is a .py file.
set -euo pipefail
cd "$(dirname "$0")"

command -v ffmpeg >/dev/null || { echo "ffmpeg missing: sudo apt install ffmpeg"; exit 1; }

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip

# cu128 wheels where there is an NVIDIA GPU (sm_89 = RTX 40-series laptop),
# CPU wheels otherwise - the CUDA build pulls ~3 GB of libraries that a machine
# without the driver can never use. run.py detects the same thing at runtime.
if command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX=https://download.pytorch.org/whl/cu128
  echo "NVIDIA GPU detected: installing CUDA wheels"
else
  TORCH_INDEX=https://download.pytorch.org/whl/cpu
  echo "no nvidia-smi: installing CPU-only wheels (transcription will be slow)"
fi
./.venv/bin/python -m pip install torch==2.9.1 torchaudio==2.9.1 \
  --index-url "$TORCH_INDEX"

# No --index-url here: these come from PyPI. They do not disturb the torch pin
# today, but nothing in the command says so, so the next step checks instead of
# trusting the comment - a resolver that quietly swapped the CUDA build for a
# PyPI wheel would otherwise show up as "CPU only" much later.
./.venv/bin/python -m pip install \
  "transformers>=4.56" accelerate faster-whisper ctranslate2 soundfile \
  "huggingface_hub[hf_xet]" \
  speechbrain scikit-learn

echo
./.venv/bin/python - "$TORCH_INDEX" <<'EOF'
import sys, torch
want_cuda = sys.argv[1].endswith("cu128")
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| CUDA:", ok,
      torch.cuda.get_device_name(0) if ok else
      "-- CPU only; run.py will detect this and is much slower")
if torch.__version__.split("+")[0] != "2.9.1":
    sys.exit(f"torch is {torch.__version__}, expected 2.9.1 - the pin was "
             f"overridden by a later install")
if want_cuda and not ok:
    sys.exit("CUDA wheels were requested but torch cannot see a GPU - the "
             "CUDA build was probably replaced by a PyPI wheel")
EOF
echo
echo "next: python run.py AUDIO   (pulls the 1.6 GB model on first run)"
