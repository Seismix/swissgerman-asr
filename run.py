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
import diarize
import transcript
from transcript import Seg

OUT_DEFAULT = pathlib.Path(__file__).resolve().parent / "out"

# Only used to tell the user *why* --names looks wrong, never to accept input.
AUDIO_EXT = {".m4a", ".mp3", ".mp4", ".wav", ".flac", ".ogg", ".opus", ".aac",
             ".webm", ".mkv", ".mov"}


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
    p.add_argument("--all", action="store_true",
                   help="run every model (not with an explicit model list)")
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
                        "int8. Default depends on --device. CTranslate2 models "
                        "only - the transformers backend takes precision from "
                        "--device and will refuse this rather than ignore it.")

    d = p.add_argument_group("speaker labels")
    d.add_argument("--diarize", action="store_true", help="label speakers")
    d.add_argument("--speakers", type=int, default=2, metavar="N",
                   help="how many speakers (default: 2)")
    d.add_argument("--names", nargs="+", metavar="NAME",
                   help="speaker names, in order of first speech, as "
                        "'--names Anna Beat' or '--names Anna,Beat'. One per "
                        "--speakers. Implies --diarize. Getting the ORDER "
                        "wrong is silent; getting the count wrong is not.")
    d.add_argument("--relabel", action="store_true",
                   help="reuse cached segments, redo only the diarization")
    d.add_argument("--min-dur", type=float, default=None, metavar="S",
                   help=f"below this a segment is too short to cluster "
                        f"(default: {diarize.MIN_DUR}). Implies --diarize.")
    d.add_argument("--margin", type=float, default=None, metavar="M",
                   help=f"how much a short segment must favour one speaker "
                        f"before its own embedding beats a neighbour's label "
                        f"(default: {diarize.MARGIN}). Implies --diarize.")
    return p


def show_models():
    w = max(len(k) for k in asr.MODELS)
    for k, (repo, _, lic, note) in asr.MODELS.items():
        print(f"{k:<{w}}  {lic:<14} {repo}\n{'':<{w}}  {note}\n")


def resolve_names(a, parser):
    """--names, flattened over commas and checked against --speakers.

    nargs="+" is greedy in two directions and both used to fail silently.
    `--names A B audio.m4a` swallows the audio path, leaving `audio` unset, so
    the run printed the model registry and exited 0 - indistinguishable from
    --list, with no transcription. And `--names=A,B`, the form CLAUDE.md
    documented, arrived as the single name "A,B", which labelled speaker 1
    "A,B" and speaker 2 "S2". Counting the names catches both.
    """
    if not a.names:
        return None
    names = [n.strip() for chunk in a.names for n in chunk.split(",") if n.strip()]
    if len(names) != a.speakers:
        hint = ""
        if a.audio is None:
            hint = ("\n--names is greedy, so it took the audio path too. Put "
                    "the audio first: run.py AUDIO --names " + ",".join(names[:a.speakers]))
        elif any(pathlib.Path(n).suffix.lower() in AUDIO_EXT for n in names):
            hint = "\none of those looks like a filename; --names takes names only"
        parser.error(f"--names got {len(names)} name(s) {names}, but --speakers "
                     f"is {a.speakers}{hint}")
    return names


def diar_opts(a):
    """Only pass knobs the user actually set, so diarize.py owns the defaults."""
    return {k: v for k, v in
            (("min_dur", a.min_dur), ("margin", a.margin)) if v is not None}


def save_segments(path, key, ident, segs):
    path.write_text(json.dumps({"model": key, "source": ident,
                                "segments": [list(s) for s in segs]},
                               ensure_ascii=False), encoding="utf-8")


def load_segments(path, key, ident):
    """Cached segments, or (None, why they cannot be reused).

    The cache was keyed on the model alone, with nothing recording which audio
    produced it. `--relabel` after transcribing a different recording therefore
    loaded the first one's segments and embedded them against the second one's
    wav: the old text, labelled by the new voices, printed with a plausible
    stats line. Out-of-range slices became zero vectors that the min_dur filter
    dropped, so nothing raised.
    """
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return None, f"it could not be read ({type(e).__name__})"
    if not isinstance(blob, dict) or "segments" not in blob:
        return None, "it predates source tracking, so it cannot be checked"
    if blob.get("model") != key:
        return None, f"it was produced by model {blob.get('model')!r}"
    was = blob.get("source") or {}
    if was != ident:
        if was.get("path") != ident["path"]:
            return None, f"it was produced from {was.get('path', '?')}"
        if was.get("clip") != ident["clip"]:
            return None, (f"it covers clip {was.get('clip') or 'the whole file'}, "
                          f"not {ident['clip'] or 'the whole file'}")
        return None, "the source file has changed since it was written"
    return [Seg(*x) for x in blob["segments"]], None


def warn_stale(out, key):
    """Say what an earlier run left behind when diarization has just failed.

    Nothing is deleted. But the old code wrote the *unlabelled* transcript to
    <key>.txt and left <key>.speakers.txt untouched, and score_speakers.py
    defaults to that exact path - so the next score silently reported the
    previous run's accuracy as though it belonged to this one.
    """
    stale = sorted(out.glob(f"{key}.speakers.*"))
    if stale:
        print("  these are from an EARLIER run and have NOT been updated:")
        for f in stale:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
            print(f"    {f}  ({when})")
    print("  nothing written for this model - fix the above and re-run")


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.list:
        show_models()
        return 0

    if a.names or a.min_dur is not None or a.margin is not None or a.relabel:
        a.diarize = True
    names = resolve_names(a, parser)
    if not a.audio:
        parser.error("an audio file is required (or --list to see the models)")
    if a.all and a.models:
        parser.error("--all runs every model; drop it, or drop the model list")
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

    # Checked before anything expensive: the point of rejecting these is to not
    # find out after a decode that the timing in _timings.tsv means something
    # other than what was asked for.
    ignored = asr.unsupported_compute_type(keys, a.compute_type)
    if ignored:
        sys.exit(f"--compute-type is a CTranslate2 setting, and {', '.join(ignored)} "
                 f"run(s) on the transformers backend where it would be accepted "
                 f"and ignored.\nDrop --compute-type, or pick CTranslate2 models "
                 f"only ({', '.join(k for k, v in asr.MODELS.items() if v[1] == 'fw')}).")

    # Parsed before the device probe, which imports torch: an unparseable
    # --clip should not cost two seconds to be told about.
    try:
        clip = asr.parse_clip(a.clip) if a.clip else None
    except ValueError as e:
        sys.exit(str(e))

    device = asr.resolve_device(a.device)
    if a.device == "auto":
        print(f"device: {asr.describe_device(device)}")

    ident = asr.source_id(src, clip)
    wav = asr.to_wav16k(src, clip)
    offset = clip[0] if clip else 0.0

    a.out.mkdir(parents=True, exist_ok=True)
    rows, failures = [], 0
    for k in keys:
        repo = asr.MODELS[k][0]
        lic = asr.MODELS[k][2]
        print(f"\n=== {k}  ({repo}) ===", flush=True)
        if asr.missing_local(k):
            print(f"  SKIPPED: {repo} not built. Run ./build_flix_ct2.py")
            rows.append((k, "SKIPPED", lic))
            failures += 1
            continue

        t0 = time.time()
        cached = a.out / f"{k}.segments.json"
        segs = None
        if a.relabel and cached.exists():
            segs, why = load_segments(cached, k, ident)
            if segs is None:
                sys.exit(f"--relabel refused: {cached} cannot be reused because "
                         f"{why}.\nRe-run without --relabel to transcribe "
                         f"{src.name} from scratch.")
            print(f"  reusing {len(segs)} cached segments")
        elif a.relabel:
            print(f"  no cached segments at {cached.name}; transcribing once")
        if segs is None:
            try:
                segs = asr.transcribe(k, wav, a.longform, offset, device,
                                      a.compute_type)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                rows.append((k, "FAILED", lic))
                failures += 1
                continue
            save_segments(cached, k, ident, segs)
        dt = time.time() - t0

        labels = None
        if a.diarize:
            t1 = time.time()
            try:
                labels, stats = diarize.label(wav, segs, a.speakers, names,
                                              device=device, offset=offset,
                                              **diar_opts(a))
                print(f"  diarize {time.time() - t1:.0f}s  {stats}")
            except Exception as e:
                print(f"  diarize FAILED: {type(e).__name__}: {e}")
                warn_stale(a.out, k)
                rows.append((k, "DIARIZE FAILED", lic))
                failures += 1
                continue

        rows.append((k, f"{dt:.0f}s", lic))
        for fmt in formats:
            text = transcript.render(segs, labels, fmt, a.merge_turns)
            dst = a.out / f"{k}{'.speakers' if labels else ''}.{fmt}"
            dst.write_text(text, encoding="utf-8")
            cwd = pathlib.Path.cwd()
            shown = dst.relative_to(cwd) if dst.is_relative_to(cwd) else dst
            print(f"  {dt:.0f}s -> {shown}")
        preview = transcript.render(segs, labels, "txt", a.merge_turns)
        print("  preview:", preview[:240].replace("\n", " | "))

    (a.out / "_timings.tsv").write_text(
        "model\tseconds\tlicense\n" + "\n".join("\t".join(r) for r in rows) + "\n",
        encoding="utf-8")
    print("\n" + "\n".join(f"{x}\t{y}\t{z}" for x, y, z in rows))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
