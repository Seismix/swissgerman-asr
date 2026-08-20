#!/usr/bin/env bash
# One-time environment setup. Needs the NVIDIA driver on the Windows host;
# WSL picks up CUDA through it, no driver install inside the distro.
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

./.venv/bin/python -m pip install \
  "transformers>=4.56" accelerate faster-whisper ctranslate2 soundfile \
  "huggingface_hub[hf_xet]" \
  speechbrain scikit-learn   # diarization; neither disturbs the torch pin

echo
./.venv/bin/python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("CUDA:", ok, torch.cuda.get_device_name(0) if ok else "-- CPU only; run.py will detect this and is much slower")
EOF
echo
echo "next: ./build_flix_ct2.sh   (one-time, ~3 GB download)"
