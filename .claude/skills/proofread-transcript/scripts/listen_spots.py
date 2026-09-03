"""Segments whose speech rate is far below the file's own median.

    python listen_spots.py [out/<key>.segments.json]

With no argument, picks the most recently written *.segments.json under out/.

A low words-per-second segment either holds a long pause or is missing speech,
and nothing here can tell which - see SKILL.md, which records how weakly this
has been validated. Prints timestamps to play, not findings.

Pure stdlib and pathlib, no shell: this runs the same on Linux, macOS and
Windows.
"""
import json
import pathlib
import statistics
import sys

# Below these the ratio measures segmentation rather than speech rate: a
# one-second backchannel is one word at 1.0 w/s and looks exactly like a drop.
MIN_DUR = 3.0
MIN_WORDS = 5
# Fraction of the file's own median under which a segment is worth playing.
FLOOR = 0.5


def pick(argv):
    if len(argv) > 1:
        p = pathlib.Path(argv[1])
        if not p.exists():
            sys.exit(f"no such file: {p}")
        return p
    # Walk up for the checkout rather than counting directories: the skill can
    # be vendored at a different depth, and a wrong count would silently find
    # nothing instead of failing usefully.
    for d in pathlib.Path(__file__).resolve().parents:
        if (d / "run.py").is_file() and (d / "out").is_dir():
            found = sorted((d / "out").glob("*.segments.json"),
                           key=lambda x: x.stat().st_mtime, reverse=True)
            if not found:
                sys.exit(f"no *.segments.json in {d / 'out'} - run run.py "
                         f"first, or pass a path")
            return found[0]
    sys.exit("could not find the checkout's out/ - pass a path to one "
             "out/<key>.segments.json")


def clock(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def main(argv):
    path = pick(argv)
    try:
        segs = json.loads(path.read_text(encoding="utf-8"))["segments"]
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"{path.name} could not be read: {type(e).__name__}")

    rows = [(s, e, e - s, len(t.split())) for s, e, t in segs if e and e > s]
    elig = [r for r in rows if r[2] >= MIN_DUR and r[3] >= MIN_WORDS]
    if len(elig) < 10:
        sys.exit(f"{path.name}: only {len(elig)} of {len(rows)} segments are "
                 f"long enough to rate - too few for a median to mean anything")

    med = statistics.median(w / d for _, _, d, w in elig)
    print(f"{path.name}: {len(elig)} of {len(rows)} segments eligible, "
          f"median {med:.2f} words/s")
    hits = [r for r in elig if r[3] / r[2] < med * FLOOR]
    for s, e, d, w in hits:
        print(f"  {clock(s)}-{clock(e)}  {d:.1f}s  {w} words  {w / d:.2f} w/s")
    print(f"  {len(hits)} span(s) below {FLOOR:g}x the median."
          if hits else "  nothing below the floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
