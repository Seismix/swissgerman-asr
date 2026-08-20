#!/usr/bin/env python3
"""Build ./flix-ct2 : Flix-AI/flix-swissgerman-full converted to CTranslate2 fp16.

    ./build_flix_ct2.py          # re-execs itself under .venv/bin/python

Why the staging step: the HF repo ships processor_config.json but NOT
preprocessor_config.json, which ct2-transformers-converter requires. It is an
upload mistake upstream. We stage the snapshot and graft that one file from
openai/whisper-large-v3 - safe because both are large-v3 at 128 mel bins
(checked against num_mel_bins in config.json, below).

This was a shell script wrapping a Python heredoc. It is Python because the work
is Python: pathlib, JSON, a hardlink-with-fallback, and a build that has to be
all-or-nothing. Doing that in bash is what left partial directories behind.
"""
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
VENV = HERE / ".venv" / "bin" / "python"
FINAL = HERE / "flix-ct2"
STAGE = HERE / "flix-src"
TMP = HERE / "flix-ct2.tmp"


def reexec_in_venv():
    """The imports below live in .venv, not in the system interpreter."""
    if pathlib.Path(sys.executable).resolve() == VENV.resolve():
        return
    if not VENV.exists():
        sys.exit("no .venv - run ./setup.sh first")
    os.execv(str(VENV), [str(VENV), str(pathlib.Path(__file__).resolve()),
                         *sys.argv[1:]])


def stage() -> pathlib.Path:
    """Copy the snapshot into STAGE and graft the missing extractor config.

    STAGE is rebuilt from scratch every time. The old script skipped any file
    that already existed, and its copy2 fallback (taken when the HF cache is on
    another filesystem) is not atomic - so an interrupted run left a truncated
    checkpoint that the next build silently reused.
    """
    from huggingface_hub import snapshot_download, hf_hub_download

    src = pathlib.Path(snapshot_download("Flix-AI/flix-swissgerman-full"))
    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir()
    for f in src.iterdir():
        if not f.is_file():
            continue
        tgt = STAGE / f.name
        try:
            os.link(f.resolve(), tgt)     # no second copy of the 3 GB checkpoint
        except OSError:
            part = tgt.with_name(tgt.name + ".part")
            shutil.copy2(f, part)
            part.replace(tgt)

    pre = pathlib.Path(hf_hub_download("openai/whisper-large-v3",
                                       "preprocessor_config.json"))
    shutil.copy2(pre, STAGE / "preprocessor_config.json")

    want = json.loads((STAGE / "config.json").read_text())["num_mel_bins"]
    got = json.loads((STAGE / "preprocessor_config.json").read_text())["feature_size"]
    if want != got:
        raise SystemExit(f"mel bin mismatch: config says {want}, extractor says {got}")
    print(f"staged -> {STAGE}  ({want} mel bins, matched)")
    return STAGE


def convert():
    """Convert into TMP, then swap. Never leave a half-built flix-ct2.

    asr.missing_local only asks whether the directory exists, so a directory
    that exists and is incomplete is worse than no directory at all: it reports
    as built and fails later as a decoder exception. The old script ran
    `rm -rf flix-ct2` *before* the converter, which is exactly the window that
    produced that state.
    """
    exe = HERE / ".venv" / "bin" / "ct2-transformers-converter"
    shutil.rmtree(TMP, ignore_errors=True)
    subprocess.run([str(exe), "--model", str(STAGE), "--output_dir", str(TMP),
                    "--copy_files", "preprocessor_config.json", "tokenizer.json",
                    "--quantization", "float16"], check=True)
    shutil.rmtree(FINAL, ignore_errors=True)
    TMP.rename(FINAL)


def main():
    reexec_in_venv()
    try:
        stage()
        convert()
    finally:
        # Runs on failure too. The old script's `rm -rf flix-src` sat after the
        # converter under `set -e`, so a failed conversion left up to 3 GB
        # behind whenever the copy2 fallback had been taken.
        shutil.rmtree(STAGE, ignore_errors=True)
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nbuilt {FINAL} - the HF cache copy can be freed with: hf cache delete")


if __name__ == "__main__":
    main()
