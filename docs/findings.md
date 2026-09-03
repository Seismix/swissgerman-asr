# Findings

Measured on 2 minutes of clean lavalier-mic dialect interview audio plus one
full 25 minute run. Two speakers, quiet room, single mono track.

> **The test audio is not in this repo and will not be.** It is a 25-minute
> two-speaker interview recorded for a school project; the person in it
> consented to be interviewed, not to be distributed. None of these numbers are
> reproducible from a clone. They are here so the conclusions can be checked
> against your own audio rather than re-derived.

**This records comparisons that have been run and closed.** They selected the
default model; `run.py` no longer compares models, and the four-way bench that
produced the early numbers is gone. Kept because the conclusions are the reason
for the current default, and re-deriving them costs GPU hours. `--model` still
points at anything else.

## Turbo vs flix, on the whole interview

The four-way bench ran on 2 minutes and chose `flix-ct2` on licence. Re-run
head-to-head on the full 25 minute interview, the quality gap it was protecting
against is not there.

| | turbo (default) | flix-ct2 |
| --- | --- | --- |
| licence | CC-BY-NC-4.0 | Apache-2.0 |
| download | 1.6 GB, already CTranslate2 | 3 GB + local conversion |
| full interview | **48 s** | 154 s |
| segments | 90 (median 33 w) | 188 (median 17 w) |
| words | 3388 | 3639 |
| turns after `--merge-turns` | 42 | 48 |
| reference content-word recall | **56.1 %** | 55.7 % |

**The 251-word gap is filler, not content.** Turbo emits 7 % fewer words but
recalls marginally *more* of the reference's content words, so what it drops is
the hesitation and repetition that the human transcriber dropped too. That is
the one result worth double-checking, because omission is this pipeline's
dangerous failure mode — measured here, it is not biting.

**Proper nouns are a wash, which is the same conclusion the four-way bench
reached.** Neither model dominates; they fail on different words.

| | reference | flix-ct2 | turbo |
| --- | --- | --- | --- |
| Bassersdorf | 1 | ✗ `Passersdorf` | ✓ |
| Schienennetz | 2 | ✓ ✓ | ✗ |
| Störung | 3 | 1 | 2 |
| Winterthur | 4 | 2 | 1 |
| Brütten | 1 | ✗ `Brütner` | ✗ `Brüttner` |
| Effretikon | 1 | ✗ `Effi` | ✗ `Effi` |
| Trassee | 3 | ✗ | ✗ |

**Where turbo is genuinely worse: turn granularity.** Its longer segments
swallow short backchannels, so `--merge-turns` yields 42 turns against the
reference's 46, where flix-ct2's finer segmentation yields 48. Over-segmenting
is the recoverable direction; a swallowed turn is not. Four turns over 25
minutes was judged an acceptable price for 3× the speed and no build step.

**Do not compare the two diarization scores directly.** Turbo scores 96.7 % raw
/ 98.6 % confidently-aligned against flix-ct2's 89.9 % / 97.2 %, but that is
mostly the *scorer*: it aligns segments to 47 reference turns, and 90 long
segments are far easier to place than 188 short ones. The honest signal is
centroid cosine distance, which is unchanged at 0.845 vs 0.846 — the voices
separate exactly as well, because the embeddings never depended on the ASR model.

## What it costs without an NVIDIA GPU

The default backend is CTranslate2, which has exactly two devices: `cuda` and
`cpu`. It rejects `rocm` and `hip`, so **an AMD card cannot run the default
model at all** — this is not a driver or a flag problem.

Measured on the hardware below, `--device cpu` against the auto-detected GPU.
Wall clock, so both include ~18 s of model load:

| | 2 min clip | full interview |
| --- | --- | --- |
| RTX 4060 Laptop | 22 s | 48 s |
| CPU, 16 threads (Ryzen 7 7840HS) | 45 s | **7 m 54 s** |

Model load dominates the short clip, which is why the gap looks like 2× there
and is 10× on the interview. Slow, not unusable.

**The CPU transcript is not the GPU transcript.** On the full interview the two
differ on 187 lines, 111 segments against 90, and 3516 words against 3470.
That is not the quantization — on the clip, `int8` and `int8_float16` on CUDA
come out byte-identical, while `cuda/int8` and `cpu/int8` do not. It is the
kernels: tiny numeric differences that beam search amplifies into different
segment boundaries, compounding over 25 minutes. Content is equivalent,
segmentation is not, and **a diarization score measured on one device does not
transfer to the other** — the scorer is sensitive to segment granularity, which
is exactly what changes here.

On AMD the GPU is not wasted entirely. The ECAPA embeddings are plain torch, and
a ROCm build runs them, so speaker labels stay on the card; `resolve_device`
returns a separate device for each side for exactly that reason. The
transformers backend is torch too, which makes
`--model Flix-AI/flix-swissgerman-full --longform` the one GPU transcription
path on a Radeon — and Apache-2.0, so the licence question goes away with it.
**Untested on real AMD hardware**: the ROCm branches were exercised by faking
`torch.version.hip`, which proves the routing and nothing about the kernels.

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
