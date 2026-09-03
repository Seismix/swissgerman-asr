---
name: proofread-transcript
description: Review a finished transcript in out/ for the error class this pipeline actually produces - proper nouns and domain terms, misheard the same way every time. Finds variant clusters and proposes a fix list for approval; it never edits without one. Use after a run of run.py, or when asked to check, clean up, or correct a transcript.
---

# Proofread a transcript

`run.py` produces text that needs a human editing pass. This skill does the part
of that pass which is visible in the text alone, and is explicit about the part
that is not.

**What it can do:** find the same name spelled four ways, spot a truncated place
name, count how many times a term needs fixing, and hand back a fix list.

**What it cannot do:** hear the recording. The dangerous failure of this
pipeline is omission, not invention — models drop filler that was really spoken,
and the sentence still reads fine without it (`docs/findings.md`). No amount of
reading catches that, and step 6's check is a weak proxy that mostly returns
nothing. Say so when you finish: the text pass does not stand in for listening.

## Portability: the rule is about writes

**Reading and searching:** use a file-reading or grep tool if your harness has
one, and a shell `grep -n` if it does not. Some harnesses ship no grep or glob
tool at all, and some actively prefer shell commands, so this is a preference
and not a constraint. Nothing here breaks by reading a file with the wrong
tool.

**Editing, in step 7:** never `sed -i`, `awk -i`, PowerShell replacements or
any other in-place shell rewrite. This one is a hard rule and it is portability,
not taste. `sed -i` takes a mandatory backup-suffix argument on macOS and
refuses one on Linux, BSD `sed` does not support `\b` at all, Windows has no
`sed`, and the transcripts are UTF-8 with umlauts on top of that. An edit tool
that replaces exact text behaves the same on all three. If you have no edit
tool, hand the user the fix list and stop — do not improvise a shell rewrite.

The one step that needs a real interpreter, step 6, ships as a `.py` file for
the same reason: a heredoc is a shell feature.

## Never send the transcript anywhere

The recording behind these transcripts is an identifiable person who consented
to be interviewed, not to be distributed. While running this skill:

- Do not web-search, fetch, or otherwise transmit any name, phrase, or segment
  taken from the transcript — not to check a spelling, not to identify a term.
- Do not write transcript content into any file outside `out/`, and never into
  a tracked file. `docs/` was scrubbed of verbatim utterances on purpose.
- Do not quote the transcript in a commit message, an issue, or a PR.

Resolve an unfamiliar term by asking the user, or by leaving it flagged and
unresolved. An unresolved flag is a fine outcome.

## 1. Find the transcript

Glob `out/*.txt` and `out/*.md`, newest first.

`<key>.speakers.txt` is the labelled run, `<key>.txt` the unlabelled one; `key`
is the model name. Prefer the labelled one when both exist — speaker attribution
helps disambiguate who says a term.

If several models are present, ask which rather than guessing. When you cannot
ask — a report-only run, or a subagent with no user — take the most recently
written one, and say in the report which file you used and what else was there,
so the choice is visible rather than silent.

If there is no transcript, do not start a run: on CPU that is minutes of work
the user did not ask for. Print the command they need and stop.

## 2. Read it, then cluster the proper nouns

Read the whole file. A 25-minute interview is around 3400 words.

`docs/findings.md` measured roughly 20 error clusters in 2822 words, nearly all
names and domain terms, with content words under 1 % wrong. So look for exactly
that, and do not audit ordinary prose:

- **Variant spellings of one entity.** Two capitalised tokens a character or two
  apart, in the same file, are almost always one name transcribed twice.
- **Truncations.** A short capitalised token that does not work as a word in the
  sentence is usually the front of a longer name.
- **Collapsed compounds.** German compounds come back as a plausible-looking
  word that is not the one meant. These often appear correctly elsewhere in the
  same file — that correct occurrence is the evidence.
- **Acronyms expanded inconsistently**, or expanded at all when they were spoken
  as letters.

## 3. Verify each candidate before proposing it

Grep the file for each candidate, with line numbers, and count the hits. The
claim worth checking is that one replacement fixes every occurrence.

Two things to read off the results:

- **Are all the hits the same entity?** If the term appears in contexts that are
  not, it is not a single find-replace, and step 4 applies.
- **Is any hit inside a longer word?** Step 7 replaces exact text and has no
  notion of a word boundary, so a candidate that also occurs as a prefix of
  something else needs surrounding context in the replacement rather than the
  bare word.

## 4. Two traps, and the bar a proposal has to clear

**Do not propose a replacement for a token with more than one sense in the
file.** `docs/findings.md` records `Strasse` appearing both as the rail term
*Trassee* and as an actual road. A global replace corrupts the transcript into
something that still reads perfectly. If the greps show mixed contexts, report
it as flagged-not-proposed with the line numbers, and let the user decide each.

**Do not guess a proper noun you cannot derive from the file.** `docs/rejected.md`
records what happened when the decoder was biased toward a term list: it fixed
three domain terms and invented a fourth word. A confident wrong noun is worse
than an obviously wrong one, because it survives proofreading. If a truncation's
target is not recoverable from context, say `target unknown` and leave it.

### The bar

"Supported by the file" needs to mean the same thing to every reviewer, or two
runs over one transcript produce two different lists. Propose a correction only
when at least one of these holds:

1. **The correct form is already in the file** for the same entity — several
   clean occurrences of `Brütten` against a single `Brüttner`.
2. **The file spells the term out.** A speaker who expands an acronym in the
   same breath has settled it: the expansion decides which letters the acronym
   should have, so a one-letter variant elsewhere is a slip, not a second term.
3. **The found token is not a word in the transcript's language, and exactly
   one real word fits the sentence** — `Gesinnennetz` is not German, and in a
   sentence about rail infrastructure only `Schienennetz` fits.

Flag, never propose, when:

- **The found token is a real word**, however oddly it reads. A season standing
  where a compass direction belongs is probably a one-letter error, and it stays
  flagged anyway: a real word in an odd place is exactly where a confident wrong
  fix survives proofreading unnoticed.
- More than one real word fits the gap.
- The only support is that the sentence would make more sense. Plausibility is
  not evidence.

This bar governs any misheard token, not only names. A collapsed compound is
judged on the same three tests as a place name — step 2 lists compounds as their
own category, and nothing above treats them differently.

**Write the report with the transcript's own terms, never this file's.** These
examples come from `docs/findings.md`, which publishes them deliberately; the
recording itself is not quoted anywhere in the tracked tree and must not start
being quoted here.

## 5. Report

Two tables. Proposed fixes first:

| line(s) | found | n | proposed | why |
| ------ | ------ | ------ | ------ | ------ |

`n` is the occurrence count from step 3, so the user can see the cost of each
decision. Then the flagged set, with no `proposed` column and a reason — mixed
senses, or target unknown.

Say how many clusters you found in how many words. Do not report a confidence
percentage; you have no way to compute one.

## 6. Where to listen

Omission is invisible in the text. The segment timings give one cheap check:
a segment whose words-per-second falls far below the file's own median either
holds a long pause or is missing speech, and nothing here can tell which.

**Read the next paragraph before quoting this at anyone.** The ratio is
meaningless on short segments — a 1-second backchannel is 1 word at 1.0 w/s and
looks like a drop — so the check floors at 3 seconds and 5 words. At that floor
it flags **0 of 106 eligible segments** on the reference interview, and 0 of 14
on the 2-minute clip. Dropping the word floor to 3 seconds alone flags 1 of 107,
which is where an earlier draft of this file got a number it should not have
quoted. So: it has never fired on a confirmed omission, and on the only audio it
has ever seen it fires on nothing at all. The mechanism is sound and unvalidated.
Treat a hit as a timestamp to play, and an empty result as the normal case
rather than a clean bill.

```bash
.venv/bin/python .claude/skills/proofread-transcript/scripts/listen_spots.py
```

On Windows the interpreter is `.venv\Scripts\python.exe` instead. Do not
shorten either to a bare `python`: on plenty of machines that name is unset and
only `python3` exists. The script takes an optional path to one
`out/<key>.segments.json` and otherwise picks the most recent. It imports
nothing outside the standard library, so if the venv is missing, the system
Python 3 will run it.

Report hits as timestamps to check, never as confirmed drops. The median is the
file's own, so it says nothing across recordings, and nothing across devices
either — CPU and GPU segment the same audio differently (`docs/findings.md`).

## 7. Apply, only on approval

Fixes are applied one cluster at a time, after the user has said yes to that
cluster — not to the list as a whole. Never batch a flagged item in with an
approved one.

Edit the file in place, replacing every occurrence of the exact text. Then grep
the old form again and confirm it returns nothing, and grep the new form and
confirm the count matches the `n` you reported. If either disagrees, something
else in the file matched — stop and say so rather than editing further.

Remember step 3's second question: the edit matches exact text, not whole words.
Replacing `Bruett` when the file also contains `Bruettikon` will corrupt the
longer word silently. When a candidate is a prefix of something else, include
enough surrounding text in the match to make it unambiguous.

Edits go to the file in `out/` only. `out/` is gitignored, and it stays that
way: never move a transcript into the tracked tree.

## Close by saying what is left

End with the omission caveat in one line, not a paragraph: the text pass is done,
the listening pass is not, and dropped words are what a read-through misses.
