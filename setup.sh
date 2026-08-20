#!/usr/bin/env bash
# One-time environment setup. Under WSL, an NVIDIA card needs the driver on the
# Windows host and nothing inside the distro; an AMD card needs a native Linux
# install, because WSL does not pass /dev/kfd through.
#
# This is the only shell script left, and it is shell because it is the one
# thing that cannot be Python: it creates the interpreter everything else runs
# in. Anything that runs *after* .venv exists is a .py file.
set -euo pipefail
cd "$(dirname "$0")"

command -v ffmpeg >/dev/null || { echo "ffmpeg missing: sudo apt install ffmpeg"; exit 1; }

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip

# Which torch build. cu128 for an NVIDIA GPU (sm_89 = RTX 40-series laptop),
# rocm6.4 for an AMD one, CPU wheels otherwise - a GPU build pulls ~3 GB of
# libraries that a machine without the driver can never use. run.py detects the
# same three cases at runtime. FLAVOUR=cpu ./setup.sh overrides the detection.
if [ -n "${FLAVOUR:-}" ]; then
  echo "FLAVOUR=$FLAVOUR taken from the environment"
elif command -v nvidia-smi >/dev/null 2>&1; then
  FLAVOUR=cu128
  echo "NVIDIA GPU detected: installing CUDA wheels"
elif command -v rocminfo >/dev/null 2>&1 || [ -e /dev/kfd ]; then
  # rocm6.4 is the newest torch index that still carries the 2.9.1 pin -
  # rocm7.0 only has 2.10. Whether it has kernels for the card is a separate
  # question, and an open one on RDNA4 (gfx1201, RX 9070); the check at the
  # bottom of this script is what answers it.
  FLAVOUR=rocm6.4
  echo "AMD GPU detected: installing ROCm wheels"
  echo "  note: CTranslate2 has no AMD backend, so the DEFAULT model still runs"
  echo "  on CPU. The GPU is used for speaker labels, and for the transformers"
  echo "  backend: run.py AUDIO --model Flix-AI/flix-swissgerman-full --longform"
else
  FLAVOUR=cpu
  echo "no NVIDIA or AMD GPU found: installing CPU-only wheels (slow)"
fi
./.venv/bin/python -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url "https://download.pytorch.org/whl/$FLAVOUR"

# No --index-url here: these come from PyPI. They do not disturb the torch pin
# today, but nothing in the command says so, so the next step checks instead of
# trusting the comment - a resolver that quietly swapped the CUDA build for a
# PyPI wheel would otherwise show up as "CPU only" much later.
./.venv/bin/python -m pip install \
  "transformers>=4.56" accelerate faster-whisper ctranslate2 soundfile \
  "huggingface_hub[hf_xet]" \
  speechbrain scikit-learn

echo
./.venv/bin/python - "$FLAVOUR" <<'EOF'
import sys, torch
want = sys.argv[1]
ok = torch.cuda.is_available()
kind = ("rocm" if getattr(torch.version, "hip", None) else "cuda") if ok else "cpu"
print("torch", torch.__version__, "|", kind,
      torch.cuda.get_device_name(0) if ok else
      "-- CPU only; run.py will detect this and is much slower")
if torch.__version__.split("+")[0] != "2.9.1":
    sys.exit(f"torch is {torch.__version__}, expected 2.9.1 - the pin was "
             f"overridden by a later install")
if want != "cpu" and not ok:
    sys.exit(f"{want} wheels were requested but torch cannot see a GPU - the "
             f"GPU build was probably replaced by a PyPI wheel")
# is_available() only proves the runtime loaded. A wheel with no compiled
# kernels for this particular card passes that check and then dies on the first
# real op, in the middle of a decode. RDNA4 is new enough for that to be a live
# risk, so spend a millisecond proving the GPU can actually multiply.
if ok:
    try:
        (torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda")).cpu()
    except Exception as e:
        sys.exit(f"torch sees the GPU but cannot compute on it: "
                 f"{type(e).__name__}: {e}\n"
                 f"On AMD this usually means the wheel has no kernels for the "
                 f"card's architecture. Try a different ROCm index, or fall "
                 f"back with: FLAVOUR=cpu ./setup.sh")
EOF
echo
echo "next: python run.py AUDIO   (pulls the 1.6 GB model on first run)"
