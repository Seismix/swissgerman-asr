# swissgerman-asr

Swiss German speech in, **Standard German** text out. Runs locally — nothing is
uploaded. 25 minutes of audio takes about 50 s on an RTX 4060 Laptop, or 8
minutes on a 16-thread CPU.

Optionally labels who is speaking.

> **The default model is CC-BY-NC-4.0** — coursework and personal research only,
> nothing billable. [licensing.md](docs/licensing.md) has the switch to an
> Apache-2.0 model.

## Setup

```bash
./setup.sh
```

Needs `ffmpeg` and Python 3, and that one line is the whole install — nothing to
configure and no build step. It picks CUDA, ROCm or CPU torch wheels for the card
it finds, then downloads the one model that card can run — 1.6 GB, or 3 GB on
AMD — into `~/.cache/huggingface`.

Two overrides exist for when that goes wrong. **Neither is needed on a normal
install:**

```bash
FLAVOUR=cpu ./setup.sh          # ignore the GPU, install CPU wheels
MODEL=Owner/Name ./setup.sh     # pull this instead: repo id, hf.co URL, or local dir
```

`MODEL` changes only what gets downloaded, not what `run.py` defaults to — so a
custom model also needs `--model` on every run.

## GPUs

`--device auto` is the default and picks the card itself. The model follows the
card, so nothing needs passing on any of them:

```bash
python run.py interview.m4a
```

AMD gets a different model because CTranslate2 has no ROCm backend — the turbo
model would be 1.6 GB that only ever runs on CPU. `--longform` is implied there.

Don't pass `--device rocm` or `cuda` with the turbo model — it is refused.

## Use

```bash
python run.py interview.m4a
```

Any format ffmpeg reads. Writes `out/<model>.txt`, one segment per line prefixed
`[HH:MM:SS]`.

Speaker labels:

```bash
python run.py interview.m4a --diarize                         # S1, S2, ...
python run.py interview.m4a --names Anna Beat                 # named
python run.py interview.m4a --names Anna Beat --merge-turns   # one block per turn
python run.py interview.m4a --relabel                         # re-cluster only, ~3 s
```

Other options:

```bash
python run.py interview.m4a --clip 5:00-7:00        # one span only
python run.py interview.m4a --format srt --format md   # also vtt, json; repeatable
python run.py interview.m4a --out results/
python run.py interview.m4a --model Owner/Name      # any HF repo, URL or local dir
python run.py interview.m4a --device cpu             # or cuda, rocm
```

`python run.py --help` lists the rest.

## Two things that bite

**`--names` takes speakers in the order they first speak,** and getting that
backwards is silent — the clustering stays right, the names just swap. Put the
audio path first, or `--names` swallows it.

**`--clip` timestamps stay absolute,** so a clipped run lines up with a full one.
An `srt` of a clipped span is therefore offset by the clip start.

## Docs

- [findings.md](docs/findings.md) — what these models actually do, and what to watch for
- [rejected.md](docs/rejected.md) — approaches tested and dropped, with the measurements
- [licensing.md](docs/licensing.md) — the default is non-commercial: what that permits, and the switch if it doesn't
- [speaker-labels.md](docs/speaker-labels.md) — diarization: how it scores, and what it still can't do

The numbers in those docs come from a 25-minute two-speaker interview recorded
for a school project. **That audio is not in this repo and will not be** — the
person in it consented to be interviewed, not to be distributed. Nothing here
is reproducible from a clone.

## Licence

The code is [MIT](LICENSE). **The weights are not.** The model `setup.sh` pulls
by default is CC-BY-NC-4.0 — non-commercial only — and that restriction is on
you, not on this repo. [licensing.md](docs/licensing.md) has the chain and the
one-flag switch to an Apache-2.0 model.
