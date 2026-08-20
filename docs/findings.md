# Findings

Measured on 2 minutes of clean lavalier-mic dialect interview audio plus one
full 25 minute run. Two speakers, quiet room, single mono track.

## Fine-tuning buys less than the model cards suggest

Four models, ~300 words, two or three words of disagreement between them.
Stock `base-v3` sits about **3 WER points** behind the Swiss German fine-tunes
(28.56 % vs 25.6 % on ASGDTS), not the chasm the cards imply.

Run the control before assuming you need a fine-tune. On clean audio the gap
may not be worth the extra 1.5 GB and the licence question.

## Published WERs do not transfer to your audio

Low leaderboard numbers largely measure agreement with a dataset's
transcription conventions, not dialect comprehension. The flix paper
(arXiv 2606.07608) makes the point itself: vanilla Whisper self-trained on the
ASGDTS *test set*, with zero Swiss German data, scores 13.88 % WER and beats
every published system.

So treat single-digit and low-teen WER claims as convention-matching. Real
published FHNW/ZHAW numbers on held-out dialect data are 14–17.5 %.

## The backend matters more than the model

`flix` and `flix-ct2` are **identical weights**. Running them produced very
different-looking output — 3 segments of up to 267 words vs 20 sentence-level
segments — and almost identical content.

The cause is the decoding path: chunked HF pipeline vs VAD-segmented
CTranslate2. Once `flix` runs with `--longform` (Whisper's own sequential
algorithm), the two agree to **99.8 %**.

If output looks wrong, check which backend produced it before blaming weights.

| | segments | longest | words | time |
|---|---|---|---|---|
| `flix` chunked | 3 | 267 w | 307 | 53 s |
| `flix --longform` | 20 | ~30 w | 305 | 42 s |
| `flix-ct2` | 20 | 33 w | 303 | 22 s |

`--longform` is faster *and* segments better. Use it whenever you use the
transformers backend.

## Errors are proper nouns, and they cluster

Across 2822 words: roughly **20 error clusters**, nearly all names and domain
terms. Content words came out under 1 % wrong. Every error was a single
find-replace that fixed all its occurrences.

Observed: place names truncated or misheard (`Effi` for Effretikon,
`Passersdorf` for Bassersdorf), compound domain terms collapsed
(`Gesinnennetz` for Schienennetz), and acronym expansions invented.

## The failure mode is omission, not hallucination

Models silently drop filler words that were really spoken. They do not invent
content. Verified against the audio in both directions: words missing from one
model's output were present in another's *and* in the recording.

Wrong words announce themselves. Dropped words do not — the sentence still
reads fine without them. **Listen for gaps, don't just proofread.**

## Traps

**Don't blind find-replace a corrected term.** On the test audio `Strasse` was
sometimes the rail term *Trassee* and sometimes an actual road. A global
replace corrupts the transcript in a way that reads perfectly fine.

**`--longform` and chunked decoding disagree on *which* words they get wrong,**
not on how many. Neither dominates. Chunked got two domain terms right that
long-form missed, and dropped a word long-form kept.

**Machine errors are safer than manual ones.** In a hand-edited pass over the
same interview, the human edit introduced two invented technical terms, dropped
a qualifier from a headline number, and inverted the sense of one statement —
all of which read as plausible prose. The ASR errors were obviously wrong.

**Keep audio and models on the Linux filesystem, not `/mnt/c`.** Cross-filesystem
I/O in WSL2 is slow enough to matter when streaming a 3 GB checkpoint. Compute
speed itself is identical to native Windows (measured: 21–22 s either way).
