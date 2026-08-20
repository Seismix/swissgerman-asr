# swissgerman-asr

Swiss German speech in, **Standard German** text out. Local, no GPU required
but much faster with one. 25 minutes of audio transcribes in about 2½ minutes
on an RTX 4060 Laptop (measured 151–154 s), plus 6 s to label the speakers.

## Setup

```bash
./setup.sh            # venv + torch (CUDA or CPU wheels, auto) + faster-whisper
./build_flix_ct2.py   # one-time, ~3 GB download, produces ./flix-ct2
```

## Run

```bash
python run.py interview.m4a                       # recommended model
python run.py interview.m4a --all                 # compare all models
python run.py interview.m4a flix --longform       # transformers backend
python run.py --list
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
| `build_flix_ct2.py` | one-time model conversion to CTranslate2 |
| `asr.py` | model registry, audio prep, the two decoding backends |
| `transcript.py` | segments, turn merging, output formats |
| `diarize.py` | speaker embeddings and clustering |
| `score_speakers.py` | dev tool: scores labels against a reference attribution |
| `setup.sh` | the only shell script: it creates the venv the rest runs in |

## Models

| key | licence | |
|---|---|---|
| `flix-ct2` | Apache-2.0 | **default** |
| `flix` | Apache-2.0 | same weights, transformers backend |
| `ct2` | CC-BY-NC-4.0 | smallest and fastest — **non-commercial only** |
| `base-v3` | Apache-2.0 | control: stock Whisper, no Swiss German fine-tune |

## Docs

- [findings.md](docs/findings.md) — what these models actually do, and what to watch for
- [rejected.md](docs/rejected.md) — approaches tested and dropped, with the measurements
- [licensing.md](docs/licensing.md) — why `ct2` is non-commercial and its clones are mislabelled
- [speaker-labels.md](docs/speaker-labels.md) — diarization: how it scores, and what it still can't do
