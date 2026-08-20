# Handover

Swiss German speech → Standard German text, running locally. Built and verified
2026-08-20. Read `README.md` for usage; this file is what a fresh agent needs
that the README doesn't say.

## State

Working and smoke-tested. Nothing is in progress, nothing is half-finished.

Five modules: `run.py` is CLI only, `asr.py` holds the model registry, audio
prep and both backends, `transcript.py` holds segments and output formats,
`diarize.py` the speaker labelling, `score_speakers.py` is a dev tool. The CLI
is argparse; an earlier version hand-parsed `--flag=value` out of `sys.argv`
and could not take positional args safely.

`setup.sh` is the only shell script, and now the only script that is not
`run.py` and its four modules. There was a `build_ct2.py` that converted a
transformers checkpoint to CTranslate2; **it was deleted on 2026-08-20** along
with the 2.9 GB `flix-ct2/` it produced, because the default model ships
already-converted and nothing needed it. It is in git history at `47d56af` if a
local conversion is ever wanted again — `git show 47d56af:build_ct2.py`.

**The model comparison is gone.** `run.py` ran a four-model registry with
`--all` and wrote `out/_timings.tsv`; that was prototype scaffolding, it picked
`flix-ct2`, and it is not worth re-running. One model per invocation now,
`--model` to change it. `docs/findings.md` keeps what the comparison concluded
— read it before re-adding a bench.

- WSL Ubuntu 24.04, `.venv` built by `setup.sh`, CUDA verified on an RTX 4060 Laptop
- **Device is auto-detected** (`--device auto`), so this runs on a machine with
  no NVIDIA GPU, slowly. `setup.sh` picks CUDA, ROCm or CPU torch wheels the
  same way, and `FLAVOUR=cpu ./setup.sh` overrides it.
- **AMD was added on 2026-08-20 and has never run on real AMD hardware.** The
  ROCm branches were exercised by setting `torch.version.hip` on a CUDA box,
  which proves the routing and nothing about the kernels. The trap it exists to
  close: a ROCm build of torch reuses the whole `torch.cuda` namespace, so
  `torch.cuda.is_available()` is True on a Radeon and the old `resolve_device`
  handed `"cuda"` to CTranslate2 — which has no AMD backend at all — one line
  after printing `device: cuda (AMD Radeon ...)`. `asr.gpu_kind()` reads
  `torch.version.hip` to tell the builds apart, and `resolve_device` now returns
  **two** devices: CPU for CTranslate2, GPU for everything torch-side, so an AMD
  box keeps GPU diarization. `run.py` rejects `--device cuda`/`rocm` on the `fw`
  backend up front rather than letting it fail at model load.
- `transformers` resolved to **5.15.1** — the major-version jump was tested, both
  the chunked and `--longform` HF paths work. Deliberately left unpinned.
- **No model is stored in the repo.** The default is pulled from HF on first
  run and cached in `~/.cache/huggingface` (1.6 GB).
- Last measured: the 25 min interview in 48 s + 3 s diarization
- **Git repo initialised 2026-08-20**, no remote. `.gitignore` excludes the
  recordings, everything derived from them, and the model weights. Before adding
  a remote, note that nothing tracked names the interviewee — keep it that way.

Test audio and all development transcripts live in `data/` (gitignored, 44 MB) —
see `data/README.md`. It contains a recording of a **named, identifiable person**
who agreed to be interviewed, not to be distributed. Use it locally; never commit
it, upload it, or send it to an external service. There is no verbatim reference
transcript, so no WER number computed from this audio would be meaningful.

## Before changing anything

**Read `docs/rejected.md` first.** It lists approaches that look like obvious
improvements, were measured, and were dropped — decoder biasing via `hotwords`,
loudness normalisation, non-Whisper models. Each entry has the numbers. Don't
re-derive them, and don't re-add them without new evidence.

The two decisions most likely to get second-guessed:

- **No glossary/hotwords.** It fixed three domain terms and invented a word.
  A confident wrong noun is worse than an obvious one.
- **The default model is CC-BY-NC**, deliberately, and this reversed on
  2026-08-20. The original four-way bench picked Apache-2.0 `flix-ct2` on
  licence grounds. Re-measured head-to-head on the *full* interview rather than
  2 minutes, the turbo model is 3× faster, needs no build step, and matches on
  content — see the top of `docs/findings.md`. This is a school project and
  CC-BY-NC permits coursework. **It does not permit billable work.** If that
  changes, `--model Flix-AI/flix-swissgerman-full --longform` is the switch —
  the transformers backend, no conversion. `docs/findings.md` measured that path
  working at 42 s on the 2 min clip against `flix-ct2`'s 22 s; it has **not**
  been re-run since the converter was deleted.

## Conventions that exist for a reason

- **Never write beside the source audio.** `to_wav16k()` caches into `cache/`.
  The source is often in a synced folder; a stray 48 MB WAV syncs and, if the
  folder is shared, shows up for other people.
- **Keep audio and models off `/mnt/c`.** Compute speed is identical to native
  Windows; cross-filesystem I/O in WSL2 is not.
- Decoding params go on `model.generation_config`, not `generate_kwargs` —
  passing both is deprecated and warns on every call.
- Console warnings were audited; the survivors are documented as harmless in
  git history and are not worth re-investigating.

## Speaker labels: done

`diarize.py` + `--names A B` on `run.py`. **98.6 %** on the full interview with
the default model (placeable segments; the raw number is 96.7 %). The old
flix-ct2 figures were 97.2 % / 89.9 %. **Those two are not comparable** — the
scorer favours coarser segmentation — and `docs/speaker-labels.md` says why and
which to quote.

Two things to know before touching it:

- **`--names` takes speakers in order of first appearance.** For
  `interview-full.m4a` that is the interviewer first, then the interviewee.
  Backwards is silent — the clustering stays correct and only the names swap.
  Names of real people belong in the command line, not in the repo: the
  interviewee consented to be recorded, not to be named in a public file.
- **`--relabel` reuses cached segments** from `out/<key>.segments.json`, 6 s
  instead of 154 s. Use it for any clustering change.
- **`diarize.embed` must subtract the clip offset**, not the first segment's
  start, before indexing the wav. Getting that wrong shifts every embedding by
  the leading silence and degrades the result silently. It has been introduced
  once already; `docs/speaker-labels.md` has the numbers it moved.

The doc's original claim that short segments must take a neighbour's label was
measured and is wrong on this audio; see the MARGIN table. Sample size is 7.

Still open: VAD segments are not speaker turns, so a segment containing both
voices sits between clusters. That is the ceiling, and resegmentation is the
only fix. Diffing against `pyannote/speaker-diarization-3.1` is the next step.

## Working style

The user corrects errors directly and expects them accepted without ceremony —
they caught a wrong hallucination claim and a wrong licence claim during the
build, both of which changed conclusions. Verify before asserting, especially
about what is or isn't in an audio file. They run commands themselves and read
output; report measurements, not impressions.
