# Licensing

Matters here because Swiss German fine-tunes are trained on dialect corpora
with restrictive terms, and those terms propagate into every derivative.

## The default is non-commercial

**`OSTswiss/whisper-large-v3-turbo-swiss-german-ct2`** — **CC-BY-NC-4.0**.
A CTranslate2 export of `Flurin17/whisper-large-v3-turbo-swiss-german`, which is
CC-BY-NC because its training data (SwissDial) is.

**Non-commercial only.** This is a school project, and coursework and personal
research are exactly what CC-BY-NC permits, so it is the default: 1.6 GB, no
build step, and 3× faster than the alternative for output that
[findings.md](findings.md) measures as equivalent.

**The moment any of this is billable, the licence is violated.** Not the moment
you publish — the moment a client pays for work the model produced. If that
becomes possible, switch to the Apache-2.0 model below and re-run; nothing else
in the pipeline changes.

The upstream card self-reports 37.96 WER on a private validation split. That is
a much worse number than flix's 25.6 %, on a non-comparable split — see
[findings.md](findings.md) on why cross-model WER comparisons don't hold, and on
why the measured difference on real audio is much smaller than that gap implies.

## The commercial escape hatch

**`Flix-AI/flix-swissgerman-full`**, **Apache-2.0**. Full fine-tune of
whisper-large-v3, from arXiv 2606.07608 (Felix Akeret, May 2026). No usage
restriction.

```bash
python run.py interview.m4a --model Flix-AI/flix-swissgerman-full --longform
```

No conversion step: that runs it on the transformers backend straight from the
hub (~3 GB). `--longform` is not optional here — without it the chunked pipeline
returns three segments of up to 267 words, which diarizes badly. See
[findings.md](findings.md).

**On an AMD GPU this is not an escape hatch, it is the default.** CTranslate2
has no ROCm backend, so a Radeon cannot run the converted model at all;
`asr.default_model()` hands that machine this one instead, `setup.sh` prefetches
it and nothing else, and `run.py` implies `--longform`. The commercial
restriction therefore never applies on AMD — the explicit `--model` above is
only needed on NVIDIA or CPU.

**This path has not been re-run since 2026-08-20**, when the CTranslate2
converter and its 2.9 GB output were deleted. The numbers above are from the
original bench, where the transformers backend on this model measured 42 s on
the 2 min clip against the converted model's 22 s — about **1.9x** the
converted flix, which is itself **3.2x** the default on the full interview
(154 s against 48 s). Chained, that puts this path at roughly **6x the
default's GPU time**: call it 5 minutes for the 25 min interview. Both legs are
measured, the product is not. Budget accordingly and verify before promising
anyone a turnaround.

If the conversion is wanted back, `build_ct2.py` is in git history:

```bash
git show 5fb0842:build_ct2.py > build_ct2.py && chmod +x build_ct2.py
```

## Mislabelled clones

At least one HF repo republishes the OSTswiss weights as Apache-2.0. **It is
wrong.** A format conversion is a derivative work; CC-BY-NC cannot be relicensed
downstream by whoever re-uploads it. `OSTswiss` is used here specifically
because it labels the restriction correctly — picking the honestly-labelled repo
does not make the restriction go away, it just means you know about it.

**Check the upstream card, not the conversion's.** Follow the chain back to
whatever was actually trained, and to the corpus it was trained on. Quantised
GGUF/MLX ports of these models carry the same restriction plus, in some cases,
an additional no-reidentification clause from the corpus.

## If you have to document it

For an AI-use journal or a methods section: name the model repo, the licence,
and the fact that it ran locally with no data sent to a third party. That last
point is the one that matters when the recording contains an identifiable
person who agreed to be interviewed, not to have their voice uploaded.
