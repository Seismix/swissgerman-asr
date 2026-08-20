#!/usr/bin/env bash
# Build ./flix-ct2 : Flix-AI/flix-swissgerman-full converted to CTranslate2 fp16.
#
# Why the staging step: the HF repo ships processor_config.json but NOT
# preprocessor_config.json, which ct2-transformers-converter requires. It is an
# upload mistake upstream. We stage the snapshot and graft that one file from
# openai/whisper-large-v3 - safe because both are large-v3 at 128 mel bins
# (checked against num_mel_bins in config.json).
set -euo pipefail
cd "$(dirname "$0")"

[ -d .venv ] || { echo "run ./setup.sh first"; exit 1; }

./.venv/bin/python - <<'EOF'
import json, os, pathlib, shutil
from huggingface_hub import snapshot_download, hf_hub_download

src = pathlib.Path(snapshot_download("Flix-AI/flix-swissgerman-full"))
dst = pathlib.Path("flix-src")
dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if not f.is_file():
        continue
    tgt = dst / f.name
    if tgt.exists():
        continue
    try:
        os.link(f.resolve(), tgt)      # no second copy of the 3 GB checkpoint
    except OSError:
        shutil.copy2(f, tgt)

pre = pathlib.Path(hf_hub_download("openai/whisper-large-v3", "preprocessor_config.json"))
shutil.copy2(pre, dst / "preprocessor_config.json")

want = json.loads((dst / "config.json").read_text())["num_mel_bins"]
got = json.loads((dst / "preprocessor_config.json").read_text())["feature_size"]
assert want == got, f"mel bin mismatch: config says {want}, extractor says {got}"
print(f"staged -> {dst}  ({want} mel bins, matched)")
EOF

rm -rf flix-ct2
./.venv/bin/ct2-transformers-converter \
  --model flix-src \
  --output_dir flix-ct2 \
  --copy_files preprocessor_config.json tokenizer.json \
  --quantization float16

rm -rf flix-src
echo
echo "built ./flix-ct2 - the HF cache copy can be freed with: hf cache delete"
