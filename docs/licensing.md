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
./build_ct2.py && python run.py interview.m4a --model ./flix-ct2
```

There is no CTranslate2 export of it on the hub, which is why `build_ct2.py`
exists: it downloads the transformers checkpoint (~3 GB) and converts locally. A
format conversion is a derivative, and Apache-2.0 permits it, so `./flix-ct2` is
Apache-2.0 too.

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
point is the one that matters when the recording contains a named, identifiable
person who agreed to be interviewed, not to have their voice uploaded.
