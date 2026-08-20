# Speaker labels

Implemented in `diarize.py`. `python run.py audio.m4a --names A B`
writes `out/<key>.speakers.txt` alongside the unlabelled `out/<key>.txt`.
Add `--merge-turns` for one paragraph per turn, which is the readable form:
188 segments become 48 turns, against 47 in the hand-written docx.

Segment → ECAPA embedding → L2-normalise → agglomerative clustering with cosine
distance at a fixed `n_clusters`. `speechbrain/spkrec-ecapa-voxceleb`,
Apache-2.0 and ungated, so `docs/licensing.md` is unaffected.

Speakers are numbered by first appearance, so `--names` takes them in the order
they speak. On `interview-full.m4a` that is `A,B` — the interviewer
opens with "Herzlich willkommen, B". Getting this backwards is silent: the
clustering is still right, only the names are swapped.

## What it scores

`score_speakers.py` scores against the speaker attribution in
`_docx_transkript.txt` (47 attributed turns). That docx is useless for WER —
edited prose, see `data/README.md` — but *who spoke* survives editing, so it
works for this and nothing else does.

| | |
| --- | --- |
| `clip.wav`, 18 segments | 18/18, all 5 turn boundaries correct |
| `interview-full.m4a`, 188 segments | **97.2 %** (140/144 placeable) |
| centroid cosine distance | 0.846 |
| cost | 6 s on top of 154 s transcription |

**The 97.2 % is the honest number; the raw figure is 89.9 %.** The docx is not
timestamp-aligned, so segments are matched to turns by monotonic DP on
content-word overlap, and that aligner is the weaker half of the measurement.
44 of 188 segments share no content words with any turn and cannot be placed at
all. Hand-checking every flagged segment against the docx text, most
"mismatches" are the aligner drifting, not the labelling — `[00:08:04]`,
`[00:13:57]`, `[00:16:32]` and `[00:23:15]` are all cases where the label is
right and the alignment is wrong. Only the two closing segments were confirmed
genuine errors, and those are fixed (below).

Don't quote a single accuracy figure from this without saying which one.

## What the original prediction got wrong

This doc used to say a sub-1.5 s segment has too little voice to embed and must
take a neighbour's label. **Noisy is not the same as useless.** Measured on the
7 short segments in the full interview, by how much closer to one centroid the
short segment's own embedding must sit before it is trusted over the neighbour:

| `MARGIN` | short from neighbour | placeable | closing 2 segments |
| --- | --- | --- | --- |
| ∞ (neighbour always) | 7 | 96.5 % | both wrong |
| 0.15 | 2 | 96.5 % | 1 of 2 right |
| **0.05** (default) | 1 | **97.2 %** | both right |
| 0.0 (embedding always) | 0 | 97.2 % | both right |

The neighbour rule fails exactly where you'd expect: a short backchannel that
*starts* a turn inherits from the long segment before it. `[00:24:57] Ja,
bitte.` (0.80 s) and `[00:24:58] Danke für die Einladung.` (1.08 s) are B
in the docx, and the nearest long segment is A's "Cool, dass du dir Zeit
genommen hast".

**This is 7 segments.** The margin is a conservative default that fixes two
confirmed errors, not a tuned parameter. Don't read the table as precision.

Also unbitten: the warning that same-mic speakers cluster closer than you'd
expect. At 0.846 these two separate cleanly. Two voices, one lavalier, a quiet
room — the easy case. Expect this to matter on worse audio.

## Still open

**Segment boundaries are still not speaker boundaries.** VAD splits on silence,
so a question and its answer can land in one segment and its embedding falls
between the clusters. Nothing here resegments, and that remains the ceiling.

**`n_clusters` is fixed.** Fine when you know it's two people. There is no
speaker-count estimation.

`--relabel` reuses `out/<key>.segments.json` and re-runs only the diarization,
which is 6 s against 154 s. Use it for anything that touches clustering.
`--min-dur` and `--margin` expose the two knobs in the table above.

**Slicing is relative to the clip, timestamps are not.** Segment times are
absolute in the source recording so that a `--clip` run lines up with a full
one, which means `diarize.embed` has to subtract the clip offset before it
indexes the wav. Subtracting anything else — the first segment's start, say —
shifts every embedding by the leading silence and quietly degrades the result
rather than failing. That bug cost 0.03 of centroid distance and three
segments when it was introduced during the module split.

## Next

Run `pyannote/speaker-diarization-3.1` over the same audio and diff. The
interesting part is not the clustering — it is seeing what resegmentation buys
on the mixed segments, which is the one thing this approach cannot fix. Note
it is gated behind terms acceptance.
