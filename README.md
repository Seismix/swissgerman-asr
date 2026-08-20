# swissgerman-asr

Swiss German speech in, **Standard German** text out. Local, no GPU required
but much faster with one. 25 minutes of audio transcribes in about 50 s on an
RTX 4060 Laptop (measured 48 s), plus 3 s to label the speakers.

> **The default model is CC-BY-NC-4.0.** Coursework and personal research are
> fine; anything a client pays for is not. [licensing.md](docs/licensing.md) has
> the one-line switch to an Apache-2.0 model.

## Setup

```bash
./setup.sh            # venv + torch (CUDA or CPU wheels, auto) + faster-whisper
```

There is no build step. The default model is already in CTranslate2 form on
HuggingFace, so the first run pulls 1.6 GB and caches it.

## Run

```bash
python run.py interview.m4a                       # the default model
python run.py interview.m4a --model ./other-ct2   # another converted model
python run.py interview.m4a --model Owner/Name --longform   # a transformers repo
```

Speaker labels:

```bash
python run.py interview.m4a --diarize             # S1, S2, ...
python run.py interview.m4a --names Anna Beat     # in order of first speech
python run.py interview.m4a --names Anna,Beat     # same thing
python run.py interview.m4a --names Anna Beat --merge-turns   # one block per turn
python run.py interview.m4a --relabel             # re-cluster, skip transcription
```

Other options:

```bash
python run.py interview.m4a --clip 5:00-7:00      # one span only
python run.py interview.m4a --format srt --format md
python run.py interview.m4a --out results/
python run.py interview.m4a --device cpu          # default is auto-detect
```

Any format ffmpeg reads. Output is `out/<model>.txt` (or `.speakers.txt` when
labelled), one segment per line, prefixed `[HH:MM:SS]` — the segment start,
counted from the beginning of the recording. `--format` also takes `srt`,
`vtt`, `json` and `md`, and is repeatable. Decoded audio caches in `cache/`,
never beside the source file.

`--names` takes speakers **in the order they first speak**. Getting it backwards
is silent: the clustering stays correct and only the names swap. Getting the
*count* wrong is not silent — one name per `--speakers`, or it refuses. Put the
audio path before `--names`, which is greedy and will otherwise swallow it.

`--clip` timestamps stay relative to the full recording, so a clipped run lines
up with a full one. That does mean an `srt` of a clipped span is offset by the
clip start.

`--device` defaults to `auto`: CUDA if there is an NVIDIA GPU, otherwise CPU,
which works and is much slower. `setup.sh` picks matching torch wheels.

## Layout

| file | |
| --- | --- |
| `run.py` | CLI and orchestration |
| `asr.py` | model resolution, audio prep, the two decoding backends |
| `transcript.py` | segments, turn merging, output formats |
| `diarize.py` | speaker embeddings and clustering |
| `score_speakers.py` | dev tool: scores labels against a reference attribution |
| `setup.sh` | the only shell script: it creates the venv the rest runs in |

## The model

`OSTswiss/whisper-large-v3-turbo-swiss-german-ct2` — **CC-BY-NC-4.0**, 1.6 GB,
already CTranslate2 so nothing is built locally. It transcribes the 25 minute
test interview in 48 s, against 154 s for a locally-converted Apache-2.0 model
that this repo no longer builds, for output [findings.md](docs/findings.md)
measures as equivalent: identical content-word recall, proper-noun errors a
wash, 7 % fewer words that are all filler.

**It is non-commercial.** That is the whole reason this is a choice rather than
an obvious default — see [licensing.md](docs/licensing.md). For anything
billable, the Apache-2.0 model runs without conversion on the transformers
backend:

```bash
python run.py interview.m4a --model Flix-AI/flix-swissgerman-full --longform
```

`--model` takes any converted directory, HF repo id, or huggingface.co URL.
`--backend` picks CTranslate2 or transformers; `auto` decides by whether the
model has `model.bin` at its root, on disk or on the hub.

**Check the licence of anything you point it at.** Several Swiss German
fine-tunes are CC-BY-NC, and at least one repo relabels one of them Apache-2.0
incorrectly.

## Docs

- [findings.md](docs/findings.md) — what these models actually do, and what to watch for
- [rejected.md](docs/rejected.md) — approaches tested and dropped, with the measurements
- [licensing.md](docs/licensing.md) — the default is non-commercial: what that permits, and the switch if it doesn't
- [speaker-labels.md](docs/speaker-labels.md) — diarization: how it scores, and what it still can't do
