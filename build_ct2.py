#!/usr/bin/env python3
"""Convert a HuggingFace Whisper model to CTranslate2, for the `fw` backend.

    ./build_ct2.py                                    # the default, -> ./flix-ct2
    ./build_ct2.py openai/whisper-large-v3            # -> ./whisper-large-v3-ct2
    ./build_ct2.py https://huggingface.co/Owner/Name  # a URL works too
    ./build_ct2.py ./some/local/checkpoint --out mine-ct2
    ./build_ct2.py --dry-run                          # print the plan, download nothing

Re-execs itself under .venv/bin/python, so ./build_ct2.py is enough.

Why the staging step: some repos ship processor_config.json but NOT
preprocessor_config.json, which ct2-transformers-converter requires - the
default model is one of them, and it is an upload mistake upstream. When it is
missing we graft it from a reference model and then check the two agree on mel
bins, because grafting a 128-bin extractor onto an 80-bin model would produce a
model that converts cleanly and transcribes garbage.

This was a shell script wrapping a Python heredoc. It is Python because the work
is Python: pathlib, JSON, a hardlink-with-fallback, and a build that has to be
all-or-nothing. Doing that in bash is what left partial directories behind.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
VENV = HERE / ".venv" / "bin" / "python"

# asr.MODELS["flix-ct2"] points at HERE/"flix-ct2", so this pair has to stay in
# step with the registry.
DEFAULT_REPO = "Flix-AI/flix-swissgerman-full"
DEFAULT_OUT = "flix-ct2"

# Where a missing preprocessor_config.json is borrowed from. large-v3 is the
# 128-mel reference; the mel check below is what makes the graft safe.
MEL_SOURCE = "openai/whisper-large-v3"


def reexec_in_venv():
    """The imports below live in .venv, not in the system interpreter."""
    if pathlib.Path(sys.executable).resolve() == VENV.resolve():
        return
    if not VENV.exists():
        sys.exit("no .venv - run ./setup.sh first")
    os.execv(str(VENV), [str(VENV), str(pathlib.Path(__file__).resolve()),
                         *sys.argv[1:]])


def parse_repo(text):
    """A local path, an 'Owner/Name' repo id, or a huggingface.co URL.

    Returns (repo, is_local). URLs are reduced to the repo id rather than
    fetched directly: huggingface_hub wants the id, and the /tree/main and
    /blob/main suffixes people paste out of the address bar are not part of it.
    """
    text = text.strip()
    local = pathlib.Path(text).expanduser()
    if local.exists():
        return local.resolve(), True
    if "://" in text:
        rest = text.split("://", 1)[1]
        host, _, path = rest.partition("/")
        if "huggingface.co" not in host:
            raise SystemExit(f"only huggingface.co URLs are understood: {text}")
        parts = [p for p in path.split("/") if p]
        for cut in ("tree", "blob", "resolve"):
            if cut in parts:
                parts = parts[:parts.index(cut)]
        if len(parts) != 2:
            raise SystemExit(f"cannot read a repo id out of: {text}")
        return "/".join(parts), False
    if text.count("/") != 1:
        raise SystemExit(f"expected 'Owner/Name', a URL, or a local path: {text}")
    return text, False


def default_out(repo, is_local):
    """flix-ct2 for the default, <name>-ct2 for anything else."""
    if not is_local and repo == DEFAULT_REPO:
        return HERE / DEFAULT_OUT
    name = pathlib.Path(repo).name if is_local else repo.split("/")[-1]
    return HERE / (name.removesuffix("-ct2") + "-ct2")


def stage(repo, is_local, stage_dir):
    """Assemble a converter-ready checkout in stage_dir.

    Rebuilt from scratch every time. The old script skipped any file that
    already existed, and its copy2 fallback (taken when the HF cache is on
    another filesystem) is not atomic - so an interrupted run left a truncated
    checkpoint that the next build silently reused.
    """
    from huggingface_hub import snapshot_download, hf_hub_download

    src = pathlib.Path(repo) if is_local else pathlib.Path(snapshot_download(repo))
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        tgt = stage_dir / f.name
        try:
            os.link(f.resolve(), tgt)     # no second copy of a 3 GB checkpoint
        except OSError:
            part = tgt.with_name(tgt.name + ".part")
            shutil.copy2(f, part)
            part.replace(tgt)

    pre = stage_dir / "preprocessor_config.json"
    if pre.exists():
        print("preprocessor_config.json present, no graft needed")
    else:
        print(f"preprocessor_config.json missing, grafting from {MEL_SOURCE}")
        shutil.copy2(pathlib.Path(hf_hub_download(MEL_SOURCE,
                                                  "preprocessor_config.json")), pre)

    cfg = json.loads((stage_dir / "config.json").read_text())
    want = cfg.get("num_mel_bins")
    got = json.loads(pre.read_text()).get("feature_size")
    if want is None:
        print(f"staged -> {stage_dir}  (no num_mel_bins in config; not a Whisper "
              f"model? converting anyway)")
    elif want != got:
        raise SystemExit(f"mel bin mismatch: config says {want}, extractor says "
                         f"{got}. Converting this would produce a model that "
                         f"transcribes noise.")
    else:
        print(f"staged -> {stage_dir}  ({want} mel bins, matched)")


def convert(stage_dir, out, quantization):
    """Convert into <out>.tmp, then swap. Never leave a half-built model.

    asr.missing_local only asks whether the directory exists, so a directory
    that exists and is incomplete is worse than no directory at all: it reports
    as built and fails later as a decoder exception. The old script ran
    `rm -rf flix-ct2` *before* the converter, which is exactly the window that
    produced that state.
    """
    exe = HERE / ".venv" / "bin" / "ct2-transformers-converter"
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    subprocess.run([str(exe), "--model", str(stage_dir), "--output_dir", str(tmp),
                    "--copy_files", "preprocessor_config.json", "tokenizer.json",
                    "--quantization", quantization], check=True)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)


def main():
    reexec_in_venv()
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"default model: {DEFAULT_REPO} -> ./{DEFAULT_OUT}")
    p.add_argument("model", nargs="?", default=DEFAULT_REPO,
                   help=f"HF repo id, huggingface.co URL, or local path "
                        f"(default: {DEFAULT_REPO})")
    p.add_argument("--out", type=pathlib.Path, default=None, metavar="DIR",
                   help="output directory (default: ./flix-ct2 for the default "
                        "model, ./<name>-ct2 otherwise)")
    p.add_argument("--quantization", default="float16", metavar="Q",
                   help="CTranslate2 weight precision (default: float16)")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be built and exit")
    a = p.parse_args()

    repo, is_local = parse_repo(a.model)
    out = (a.out if a.out is not None else default_out(repo, is_local)).resolve()
    stage_dir = out.with_name(out.name + ".src")

    # A local source whose derived name collides with its own directory would
    # be deleted by the swap at the end of convert(). ./build_ct2.py ./flix-ct2
    # derives exactly that.
    if is_local and (out == pathlib.Path(repo) or stage_dir == pathlib.Path(repo)):
        raise SystemExit(f"--out would overwrite the source directory {repo}.\n"
                         f"Pass a different --out.")

    print(f"model  {repo}{' (local)' if is_local else ''}")
    print(f"out    {out}")
    print(f"quant  {a.quantization}")
    if a.dry_run:
        print("\ndry run, nothing downloaded or written")
        return
    if out.exists():
        print(f"note: {out.name} exists and will be replaced once the new build "
              f"succeeds")

    try:
        stage(repo, is_local, stage_dir)
        convert(stage_dir, out, a.quantization)
    finally:
        # Runs on failure too. The old script's `rm -rf flix-src` sat after the
        # converter under `set -e`, so a failed conversion left up to 3 GB
        # behind whenever the copy2 fallback had been taken.
        shutil.rmtree(stage_dir, ignore_errors=True)
        shutil.rmtree(out.with_name(out.name + ".tmp"), ignore_errors=True)
    print(f"\nbuilt {out}")
    if not is_local:
        print("the HF cache copy can be freed with: hf cache delete")
    if out.name != DEFAULT_OUT:
        print(f"\nto use it, add an entry to MODELS in asr.py pointing at "
              f"{out}\nwith backend \"fw\".")


if __name__ == "__main__":
    main()
