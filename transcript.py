"""Segments and the formats they are written out in.

A Seg is (start, end, text) in seconds from the start of the source file — when
--clip is used the offset is already applied, so timestamps always refer to the
original recording rather than the trimmed span.
"""
import json
from typing import NamedTuple


class Seg(NamedTuple):
    start: float
    end: float | None
    text: str


def clock(t, ms=False, sep=","):
    """Seconds -> HH:MM:SS, optionally with subtitle-style milliseconds.

    Rounded to milliseconds *once*, up front. Flooring the seconds field and
    rounding the millisecond field separately lets the two disagree across a
    whole second: 1.9998 came out as `00:00:01,1000`, a four-digit millisecond
    field that a subtitle parser rejects.
    """
    t = max(0.0, float(t or 0))
    if not ms:
        # Floored, deliberately: a displayed segment timestamp names the second
        # the segment starts in, and rounding it would move every timestamp
        # already quoted in docs/ by up to a second.
        h, m, s, _ = _hms(int(t) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}"
    h, m, s, msec = _hms(int(round(t * 1000)))
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{msec:03d}"


def _hms(total_ms):
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return h, m, s, msec


def merge_turns(segs, labels=None):
    """Collapse consecutive segments by the same speaker into one turn.

    Without labels there is no speaker to merge on, so this is a no-op and the
    caller gets its segments back unchanged.
    """
    if labels is None:
        return [(s, None) for s in segs]
    out = []
    for seg, lab in zip(segs, labels):
        if out and out[-1][1] == lab:
            prev = out[-1][0]
            out[-1] = (Seg(prev.start, seg.end,
                           f"{prev.text} {seg.text}".strip()), lab)
        else:
            out.append((seg, lab))
    return out


def _pairs(segs, labels):
    return list(zip(segs, labels if labels is not None else [None] * len(segs)))


def as_txt(segs, labels=None):
    return "\n".join(
        f"[{clock(s.start)}] " + (f"{lab}: " if lab else "") + s.text
        for s, lab in _pairs(segs, labels))


def as_md(segs, labels=None):
    lines = []
    for s, lab in _pairs(segs, labels):
        head = f"**{lab}** " if lab else ""
        lines.append(f"{head}`[{clock(s.start)}]`\n\n{s.text}\n")
    return "\n".join(lines)


def as_json(segs, labels=None):
    return json.dumps(
        [{"start": s.start, "end": s.end, "speaker": lab, "text": s.text}
         for s, lab in _pairs(segs, labels)], ensure_ascii=False, indent=2)


def _cues(segs, labels, sep):
    """Subtitle cues. A segment with no end (the HF backend's last chunk) is
    given a nominal 2 s rather than dropped."""
    for i, (s, lab) in enumerate(_pairs(segs, labels), 1):
        end = s.end if s.end is not None else s.start + 2.0
        text = (f"{lab}: " if lab else "") + s.text
        yield i, clock(s.start, ms=True, sep=sep), clock(end, ms=True, sep=sep), text


def as_srt(segs, labels=None):
    return "\n".join(f"{i}\n{a} --> {b}\n{t}\n"
                     for i, a, b, t in _cues(segs, labels, ","))


def as_vtt(segs, labels=None):
    body = "\n".join(f"{a} --> {b}\n{t}\n"
                     for _, a, b, t in _cues(segs, labels, "."))
    return "WEBVTT\n\n" + body


FORMATS = {"txt": as_txt, "md": as_md, "json": as_json,
           "srt": as_srt, "vtt": as_vtt}


def render(segs, labels=None, fmt="txt", merge=False):
    if merge:
        pairs = merge_turns(segs, labels)
        segs = [s for s, _ in pairs]
        labels = None if labels is None else [l for _, l in pairs]
    return FORMATS[fmt](segs, labels)
