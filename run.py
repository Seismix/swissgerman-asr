"""Swiss German ASR bench - Swiss German speech in, Standard German text out.

    python run.py audio.m4a                        # recommended model
    python run.py audio.m4a --all                  # every model, for comparison
    python run.py audio.m4a flix --longform        # HF backend, long-form decoding
    python run.py audio.m4a --names Anna Beat      # transcribe and label speakers
    python run.py audio.m4a --clip 5:00-7:00       # one span only
    python run.py --list

Transcripts land in out/<key>.txt, timings in out/_timings.tsv. Decoded audio
is cached in cache/ so the source directory is never written to.
"""
import argparse
import json
import os
import pathlib
import sys
import time

# hf_transfer is gone; its successor is Xet. Merely *having* the old var set
# makes huggingface_hub emit a FutureWarning, so pop it as well as not set it.
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import asr
import transcript
from transcript import Seg

OUT_DEFAULT = pathlib.Path(__file__).resolve().parent / "out"


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="model keys: " + ", ".join(asr.MODELS))
    p.add_argument("audio", nargs="?", help="any format ffmpeg reads")
    p.add_argument("models", nargs="*", default=None,
                   help=f"model keys (default: {asr.DEFAULT})")
    p.add_argument("--list", action="store_true",
                   help="show the model registry and exit")
    p.add_argument("--all", action="store_true", help="run every model")
    p.add_argument("--longform", action="store_true",
                   help="HF backend: Whisper's sequential algorithm")
    p.add_argument("--clip", metavar="START-END",
                   help="transcribe one span only, e.g. 5:00-7:00. Timestamps "
                        "stay relative to the full recording.")
    p.add_argument("--out", type=pathlib.Path, default=OUT_DEFAULT,
                   metavar="DIR", help="output directory (default: out/)")
    p.add_argument("--format", dest="formats", action="append",
                   choices=sorted(transcript.FORMATS), metavar="FMT",
                   help="output format, repeatable: "
                        + ", ".join(sorted(transcript.FORMATS)))
    p.add_argument("--merge-turns", action="store_true",
                   help="one paragraph per speaker turn (needs speaker labels)")

    m = p.add_argument_group("machine")
    m.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="auto detects CUDA and falls back to CPU (default)")
    m.add_argument("--compute-type", default=None, metavar="T",
                   help="CTranslate2 precision, e.g. int8_float16, float16, "
                        "int8. Default depends on --device.")

    d = p.add_argument_group("speaker labels")
    d.add_argument("--diarize", action="store_true", help="label speakers")
    d.add_argument("--speakers", type=int, default=2, metavar="N",
                   help="how many speakers (default: 2)")
    d.add_argument("--names", nargs="+", metavar="NAME",
                   help="speaker names, in order of first speech. Implies "
                        "--diarize. Getting the order wrong is silent.")
    d.add_argument("--relabel", action="store_true",
                   help="reuse cached segments, redo only the diarization")
    d.add_argument("--min-dur", type=float, default=None, metavar="S",
                   help=f"below this a segment is too short to cluster "
                        f"(default: {__import__('diarize').MIN_DUR})")
    d.add_argument("--margin", type=float, default=None, metavar="M",
                   help="how much a short segment must favour one speaker "
                        "before its own embedding beats a neighbour's label")
    return p


def show_models():
    w = max(len(k) for k in asr.MODELS)
    for k, (repo, _, lic, note) in asr.MODELS.items():
        print(f"{k:<{w}}  {lic:<14} {repo}\n{'':<{w}}  {note}\n")


def diar_opts(a):
    """Only pass knobs the user actually set, so diarize.py owns the defaults."""
    return {k: v for k, v in
            (("min_dur", a.min_dur), ("margin", a.margin)) if v is not None}


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.list or not a.audio:
        show_models()
        return 0

    if a.names or a.min_dur is not None or a.margin is not None or a.relabel:
        a.diarize = True
    if a.merge_turns and not a.diarize:
        print("note: --merge-turns has nothing to merge without speaker labels")
    formats = a.formats or ["txt"]

    src = pathlib.Path(a.audio).expanduser()
    if not src.exists():
        sys.exit(f"no such file: {src}")
    keys = a.models or (list(asr.MODELS) if a.all else [asr.DEFAULT])
    unknown = [k for k in keys if k not in asr.MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(unknown)} (try --list)")

    device = asr.resolve_device(a.device)
    if a.device == "auto":
        print(f"device: {asr.describe_device(device)}")

    try:
        clip = asr.parse_clip(a.clip) if a.clip else None
    except ValueError as e:
        sys.exit(str(e))
    wav = asr.to_wav16k(src, clip)
    offset = clip[0] if clip else 0.0

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in keys:
        repo = asr.MODELS[k][0]
        print(f"\n=== {k}  ({repo}) ===", flush=True)
        if asr.missing_local(k):
            print(f"  SKIPPED: {repo} not built. Run ./build_flix_ct2.sh")
            rows.append((k, "SKIPPED", asr.MODELS[k][2]))
            continue

        t0 = time.time()
        cached = a.out / f"{k}.segments.json"
        if a.relabel and cached.exists():
            segs = [Seg(*x) for x in json.loads(cached.read_text())]
            print(f"  reusing {len(segs)} cached segments")
        else:
            try:
                segs = asr.transcribe(k, wav, a.longform, offset, device,
                                      a.compute_type)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                rows.append((k, "FAILED", asr.MODELS[k][2]))
                continue
            cached.write_text(json.dumps(segs), encoding="utf-8")
        dt = time.time() - t0
        rows.append((k, f"{dt:.0f}s", asr.MODELS[k][2]))

        labels = None
        if a.diarize:
            import diarize
            t1 = time.time()
            try:
                labels, stats = diarize.label(wav, segs, a.speakers, a.names,
                                              device=device, offset=offset,
                                              **diar_opts(a))
                print(f"  diarize {time.time() - t1:.0f}s  {stats}")
            except Exception as e:
                print(f"  diarize FAILED: {type(e).__name__}: {e}")

        for fmt in formats:
            text = transcript.render(segs, labels, fmt, a.merge_turns)
            suffix = ".speakers" if labels else ""
            dst = a.out / f"{k}{suffix}.{fmt}"
            dst.write_text(text, encoding="utf-8")
            print(f"  {dt:.0f}s -> {dst.relative_to(pathlib.Path.cwd())}"
                  if dst.is_relative_to(pathlib.Path.cwd()) else f"  -> {dst}")
        preview = transcript.render(segs, labels, "txt", a.merge_turns)
        print("  preview:", preview[:240].replace("\n", " | "))

    (a.out / "_timings.tsv").write_text(
        "model\tseconds\tlicense\n" + "\n".join("\t".join(r) for r in rows) + "\n",
        encoding="utf-8")
    print("\n" + "\n".join(f"{x}\t{y}\t{z}" for x, y, z in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
