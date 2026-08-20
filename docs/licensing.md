# Licensing

Matters here because Swiss German fine-tunes are trained on dialect corpora
with restrictive terms, and those terms propagate into every derivative.

## The clean option

**`flix` / `flix-ct2`** — `Flix-AI/flix-swissgerman-full`, **Apache-2.0**.
Full fine-tune of whisper-large-v3, from arXiv 2606.07608 (Felix Akeret,
May 2026). No usage restriction. This is why it's the default despite `ct2`
being smaller and faster.

`flix-ct2` is that model converted to CTranslate2 by `build_flix_ct2.py`.
A format conversion is a derivative, and Apache-2.0 permits it — so the
conversion is Apache-2.0 too.

## The restricted option

**`ct2`** — `OSTswiss/whisper-large-v3-turbo-swiss-german-ct2`,
**CC-BY-NC-4.0**. A CTranslate2 export of `Flurin17/whisper-large-v3-turbo-swiss-german`,
which is CC-BY-NC because its training data (SwissDial) is.

**Non-commercial only.** Fine for coursework and personal research. Not fine
for anything a client pays for.

The upstream card self-reports 37.96 WER on a private validation split. That
is a much worse number than flix's 25.6 %, on a non-comparable split — see
[findings.md](findings.md) on why cross-model WER comparisons don't hold.

## Mislabelled clones

At least one HF repo republishes the same converted weights as Apache-2.0.
**It is wrong.** A format conversion is a derivative work; CC-BY-NC cannot be
relicensed downstream by whoever re-uploads it. `OSTswiss` is used here
specifically because it labels the restriction correctly.

**Check the upstream card, not the conversion's.** Follow the chain back to
whatever was actually trained, and to the corpus it was trained on. Quantised
GGUF/MLX ports of these models carry the same restriction plus, in some cases,
an additional no-reidentification clause from the corpus.

## If you have to document it

For an AI-use journal or a methods section: name the model repo, the licence,
and the fact that it ran locally with no data sent to a third party. That last
point is the one that matters when the recording contains a named, identifiable
person who agreed to be interviewed, not to have their voice uploaded.
