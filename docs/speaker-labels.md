# Speaker labels

Implemented in `diarize.py`. `python run.py audio.m4a --names Anna Beat`
writes `out/<key>.speakers.txt` alongside the unlabelled `out/<key>.txt`.
Add `--merge-turns` for one paragraph per turn, which is the readable form: on
the default model 90 segments become 42 turns, against 46 in the hand-written
reference.

Segment → ECAPA embedding → L2-normalise → agglomerative clustering with cosine
distance at a fixed `n_clusters`. `speechbrain/spkrec-ecapa-voxceleb`,
Apache-2.0 and ungated, so `docs/licensing.md` is unaffected.

Speakers are numbered by first appearance, so `--names` takes them in the order
they speak. Getting this backwards is silent: the clustering is still right,
only the names are swapped.

> The numbers below come from a 25-minute two-speaker interview recorded for a
> school project. **That audio is not in this repo and will not be** — the
> person in it consented to be interviewed, not to be distributed. Nothing here
> is reproducible from a clone; it is recorded so the conclusions can be
> checked against your own audio rather than re-derived.

## What it scores

`score_speakers.py` scores a labelled transcript against the speaker
attribution in a reference transcript — any file with `Name: text` lines. On
the test interview that reference was a hand-written transcript with 47
attributed turns. It is useless for WER, being edited prose, but *who spoke*
survives editing, so it works for this and nothing else does.

By ASR model — **the diarization is identical code and the same embeddings;
only the segmentation it is handed differs**:

| | default (turbo) | flix-ct2 |
| --- | --- | --- |
| segments | 90 | 188 |
| confidently aligned | **98.6 %** (71/72) | 97.2 % (140/144) |
| all segments | 96.7 % (87/90) | 89.9 % (169/188) |
| unplaceable | 18 | 44 |
| centroid cosine distance | 0.845 | 0.846 |
| cost | 3 s on top of 48 s | 6 s on top of 154 s |

**Those two columns are not comparable and the higher number is not an
improvement.** The scorer aligns segments to 47 reference turns, and 90 long
segments are far easier to place than 188 short ones, so the model that
segments more coarsely scores better without labelling anything better. The
honest cross-model signal is centroid cosine distance, unchanged at 0.845 vs
0.846 — the embeddings never saw the ASR model's output.

**Within one model, quote the confidently-aligned figure; the raw one is
pessimistic.** The reference is not timestamp-aligned, so segments are matched
to turns by monotonic DP on content-word overlap, and that aligner is the
weaker half of the measurement. Hand-checking every flagged segment, most
"mismatches" are the aligner drifting, not the labelling. On flix-ct2, four of
the six were cases where the label is right and the alignment is wrong; only
the two closing segments were confirmed genuine errors, and those are fixed
(below). All three of turbo's mismatches are one-word backchannels that the
aligner places at conf 0.00–0.40.

Don't quote a single accuracy figure from this without saying which one, and
which model produced the segments.

The rest of this section is measured on **flix-ct2**, whose finer segmentation
is what exposed these effects; turbo's 90 coarser segments show none of them
(0 of its mismatches rest on a single shared word, and it has 2 short segments
against flix-ct2's 7). The mechanism is unchanged, so it will come back on any
model that segments finely.

The `conf >= 0.30` gate is a *ratio* — shared content words over the segment's
content words — with no floor on how many words that is. A segment with one
content word that happens to occur in the aligned turn scores 1.00 on the
strength of a single coincidence. 13 of the 144 placeable segments are in that
position, and **all four remaining mismatches are among them** — at conf 1.00,
0.50, 0.50 and 0.33, each resting on exactly one shared word.

Require two shared content words and the figure is 131/131. **Do not quote
that as an accuracy** — the floor was picked after seeing which segments failed,
which is how you tune a number into meaninglessness. What it is good for is
direction: the missing floor makes 97.2 % *pessimistic*, not optimistic, and it
corroborates mechanically what hand-checking said, that these four are the
aligner drifting rather than the labelling. `score_speakers.py` prints the
shared-word count per mismatch so this is visible instead of inferred.

## What the original prediction got wrong

This doc used to say a sub-1.5 s segment has too little voice to embed and must
take a neighbour's label. **Noisy is not the same as useless.** Measured on the
7 short segments flix-ct2 produces on the full interview (turbo produces 2, and
takes neither from a neighbour, so this table is not reachable on the default),
by how much closer to one centroid the short segment's own embedding must sit
before it is trusted over the neighbour:

| `MARGIN` | short from neighbour | placeable | closing 2 segments |
| --- | --- | --- | --- |
| ∞ (neighbour always) | 7 | 96.5 % | both wrong |
| 0.15 | 2 | 96.5 % | 1 of 2 right |
| **0.05** (default) | 1 | **97.2 %** | both right |
| 0.0 (embedding always) | 0 | 97.2 % | both right |

The neighbour rule fails exactly where you'd expect: a short backchannel that
*starts* a turn inherits from the long segment before it. The two closing
segments are both under 1.1 s, both the second speaker, and the nearest long
segment is the first speaker signing off.

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
