# Handover

Swiss German speech → Standard German text, running locally. Built and verified
2026-08-20. Read `README.md` for usage; this file is what a fresh agent needs
that the README doesn't say.

Working and smoke-tested. Nothing is in progress, nothing is half-finished.

## Layout

Five modules: `run.py` is CLI only (argparse), `asr.py` holds the model
registry, audio prep and both backends, `transcript.py` holds segments and
output formats, `diarize.py` the speaker labelling, `score_speakers.py` is a dev
tool. `setup.sh` is the only shell script, and it is shell because it builds the
interpreter everything else runs in.

One model per invocation, `--model` to change it. An earlier `run.py` ran a
four-model registry with `--all`; that was prototype scaffolding and is gone.
`docs/findings.md` keeps what the comparison concluded — read it before
re-adding a bench.

## Devices and models

- **Device is auto-detected** (`--device auto`), so this runs with no GPU,
  slowly. `setup.sh` picks CUDA, ROCm or CPU torch wheels the same way, and
  `FLAVOUR=cpu ./setup.sh` overrides it. Verified on WSL Ubuntu 24.04 with an
  RTX 4060 Laptop.
- **AMD has never run on real AMD hardware.** The ROCm branches were exercised
  by setting `torch.version.hip` on a CUDA box, which proves the routing and
  nothing about the kernels. The trap it exists to close: a ROCm build of torch
  reuses the whole `torch.cuda` namespace, so `torch.cuda.is_available()` is
  True on a Radeon and the old `resolve_device` handed `"cuda"` to CTranslate2 —
  which has no AMD backend at all — one line after printing
  `device: cuda (AMD Radeon ...)`. `asr.gpu_kind()` reads `torch.version.hip` to
  tell the builds apart, and `resolve_device` returns **two** devices: CPU for
  CTranslate2, GPU for everything torch-side, so an AMD box keeps GPU
  diarization. `run.py` rejects `--device cuda`/`rocm` on the `fw` backend up
  front rather than at model load.
- **No model is stored in the repo.** `setup.sh` ends by running
  `run.py --prefetch`, which pulls **one** model into `~/.cache/huggingface` —
  the one this machine's GPU can run. `MODEL=Owner/Name ./setup.sh` overrides
  what gets pulled but not what `run.py` defaults to; that asymmetry is
  deliberate and the script says so on the way out.
- **The default model depends on the card.** `asr.default_model()` returns
  `ROCM_MODEL` (`Flix-AI/flix-swissgerman-full`, Apache-2.0, ~3 GB,
  transformers) on a ROCm box and `DEFAULT_MODEL` (the CC-BY-NC turbo CT2 model,
  1.6 GB) everywhere else, so no machine downloads weights its GPU cannot use.
  `run.py` implies `--longform` when it reaches the AMD model *as a default*; an
  explicit `--model` never gets it. `parse_model_spec(None)` therefore imports
  torch, which is why the `--clip` parse happens before it rather than after.
- `transformers` resolved to **5.15.1** — the major-version jump was tested,
  both the chunked and `--longform` HF paths work. Deliberately left unpinned.
- Last measured: 25 min of interview audio in 48 s + 3 s diarization, or 7 m
  54 s on 16 CPU threads. **CPU and GPU do not produce the same transcript** —
  111 segments against 90. It is the kernels, not the quantization (`int8` and
  `int8_float16` on CUDA are byte-identical), and it means a diarization score
  measured on one device does not transfer to the other.

A `build_ct2.py` converted a transformers checkpoint to CTranslate2; it was
deleted on 2026-08-20 with the 2.9 GB output it produced, because the default
model ships already-converted. Recover with `git show 1fa032a:build_ct2.py`.

## Before changing anything

**Read `docs/rejected.md` first.** It lists approaches that look like obvious
improvements, were measured, and were dropped — decoder biasing via `hotwords`,
loudness normalisation, non-Whisper models. Each entry has the numbers.

The two decisions most likely to get second-guessed:

- **No glossary/hotwords.** It fixed three domain terms and invented a word.
  A confident wrong noun is worse than an obvious one.
- **The default model is CC-BY-NC**, deliberately. The original bench picked
  Apache-2.0 `flix-ct2` on licence grounds; re-measured on the *full* interview
  rather than 2 minutes, the turbo model is 3× faster, needs no build step, and
  matches on content. It permits coursework, **not billable work**. The switch
  is `--model Flix-AI/flix-swissgerman-full --longform` — the transformers
  backend, no conversion. That path measured 42 s on the 2 min clip against
  `flix-ct2`'s 22 s; it has **not** been re-run since the converter was deleted.

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
- Report measurements, not impressions. Verify before asserting, especially
  about what is or isn't in an audio file.

## Test data

`data/` is gitignored and **not in the public repo**. Locally it holds a
recording of an identifiable person who agreed to be interviewed, not to be
distributed: use it locally, never commit it, upload it, or send it to an
external service. There is no verbatim reference transcript, so no WER number
computed from it would be meaningful.

Nothing tracked names the interviewee or quotes the recording — the docs were
scrubbed of verbatim utterances before the repo went public. Keep it that way.

## Speaker labels: done

`diarize.py` + `--names A B` on `run.py`. **98.6 %** on the full interview with
the default model (placeable segments; the raw number is 96.7 %). The old
flix-ct2 figures were 97.2 % / 89.9 %. **Those two are not comparable** — the
scorer favours coarser segmentation — and `docs/speaker-labels.md` says why and
which to quote.

- **`--names` takes speakers in order of first appearance.** Backwards is
  silent — the clustering stays correct and only the names swap. Names of real
  people belong in the command line, not in the repo.
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
