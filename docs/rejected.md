# Tried and rejected

Things that look like obvious improvements and are not. Each was measured
before being dropped, so don't re-derive them.

## Glossary / decoder biasing

**What:** feed a list of proper nouns and domain terms to the decoder so it
spells them your way.

**How, correctly:** faster-whisper's `hotwords` parameter. It is prepended to
the decode prompt of *every* window.

**Not `initial_prompt`** — that only primes the first window. With
`condition_on_previous_text=False` (which this pipeline uses, to stop error
propagation), its influence is dead after ~30 s. Measured: the opening line
changed, the same error 55 s later did not.

**Result with `hotwords`:** three domain terms fixed on the test clip. Also:

- segmentation coarsened — 20 segments became 12, including one 80-word block
- the first four words of the audio were dropped
- the decoder **invented a word**, bending a real one toward a glossary entry
  (`Güterverkehr` → `Güterverkehrsleiter`, because the list contained
  `Zugverkehrsleiter` and `Fahrdienstleiter`)

Narrowing the list to proper nouns only, on the theory that the job titles
caused the bleed, made it **worse** — the term it had fixed reverted, and the
invented word survived in another form.

**Verdict: not enabled.** A confident wrong noun is worse than an obviously
wrong one, because you won't catch it proofreading.

## Loudness normalisation

**What:** EBU R128 (`loudnorm=I=-16:TP=-1.5:LRA=11`) before transcription, on
the theory that a quiet speaker transcribes worse.

Whisper's log-mel feature extractor self-normalises, so pure gain changes
largely wash out before they reach the model. The only plausible upside is
loudnorm's dynamic-range compression lifting the quieter of two speakers.

Never measured a benefit. Not in the pipeline. If you want to test it, the
one-liner is above and the comparison costs one 22 s run.

## Non-Whisper models

Canary-1B-v2, Parakeet-TDT-0.6B-v3, Voxtral Transcribe 2 and Qwen3-ASR all top
2026 German leaderboards. None of them does dialect → Standard German
normalisation, which is the actual task here — they transcribe what they hear,
and Swiss German written phonetically is not usable German text.

No usable Swiss German fine-tune of any of them exists. The Voxtral Swiss
German repos on HF are auto-generated empty cards with zero downloads.

**Whisper fine-tunes are currently the only real option.** Re-check
periodically; this is the part of the analysis most likely to go stale.

## Apertus

Swiss-made, audio input, right nationality — wrong tool. The audio capability
is explicitly labelled experimental, there is no ASR benchmark, and there is no
Swiss German dialect claim anywhere in the release. Not an ASR model.

## Paid transcription services

Swiss providers are the credible alternative, and the good ones are genuinely
good: broadcast-trained models, speaker separation, all data processed and
stored in Switzerland, usually a student rate. List price is around
**CHF 1.00/min**, dropping to roughly CHF 0.13 in bulk — not the CHF 0.10 that
gets quoted around.

Distrust the accuracy claims, including the favourable ones. One published
test of a Swiss service reported spending *20 minutes* correcting a 2-minute
transcript, against 17 minutes to type it from scratch. The comparison table
that circulates most widely was published by a vendor that appears in it.
Measure on your own audio before paying for anything.

The local pipeline is free, offline, and needs no data-processing agreement
for an identifiable interviewee. That last part is usually the deciding
factor.
