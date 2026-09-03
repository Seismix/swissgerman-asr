"""Speaker labels: one ECAPA embedding per segment, then cluster.

The approach, its measurements and its remaining ceiling are in
docs/speaker-labels.md. In short: VAD segments are not speaker turns, so this
works well on clean turn-taking and degrades on segments where both people
talk. It is the cheap baseline a resegmenting pipeline has to beat.
"""
import pathlib

import numpy as np

# spkrec-ecapa-voxceleb is Apache-2.0 and ungated. pyannote/embedding is gated
# behind terms acceptance, which docs/licensing.md would then have to explain.
ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
SAVEDIR = pathlib.Path(__file__).resolve().parent / "models" / "ecapa"

# Below this a segment - a one-word backchannel, typically - is too short to
# cluster on.
MIN_DUR = 1.5

# How much closer to one centroid such a segment must sit before its own noisy
# embedding is trusted over a neighbour's label. 0 always trusts it, a huge
# value never does. Measured, not guessed - see the table in the doc.
MARGIN = 0.05


def _spans(segments, total):
    """Segment (start, end), with the HF backend's trailing None filled in."""
    out = []
    for i, seg in enumerate(segments):
        end = seg[1]
        if end is None:
            end = segments[i + 1][0] if i + 1 < len(segments) else total
        out.append((seg[0], min(end, total)))
    return out


def embed(wav, segments, device="cuda", offset=0.0):
    """One L2-normalised 192-d vector per segment, plus the spans used."""
    import soundfile as sf
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # speechbrain 1.1 cannot parse a bare "cuda" and falls back with a warning.
    # "cuda" is also what a ROCm build of torch calls a Radeon, so this is the
    # GPU path on AMD too - unlike asr.run_fw, which has no AMD backend at all.
    dev = "cuda:0" if device == "cuda" else device
    enc = EncoderClassifier.from_hparams(source=ECAPA, savedir=str(SAVEDIR),
                                         run_opts={"device": dev})
    # Segment times are absolute in the source recording, but the wav may be a
    # clip of it starting at `offset`. Subtracting anything else - the first
    # segment's start, say - silently shifts every slice by the leading silence.
    spans = _spans(segments, offset + len(audio) / sr)
    vecs = []
    for start, end in spans:
        a, b = int((start - offset) * sr), int((end - offset) * sr)
        chunk = audio[max(0, a):max(0, b)]
        if len(chunk) < sr // 10:          # <100 ms: nothing to encode
            vecs.append(np.zeros(192, dtype=np.float32))
            continue
        with torch.no_grad():
            v = enc.encode_batch(torch.from_numpy(chunk).unsqueeze(0))
        vecs.append(v.squeeze().cpu().numpy())
    x = np.vstack(vecs)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1, n), spans


def label(wav, segments, n_speakers=2, names=None,
          min_dur=MIN_DUR, margin=MARGIN, device="cuda", offset=0.0):
    """Returns (labels, stats). One label per segment, in order."""
    from sklearn.cluster import AgglomerativeClustering

    x, spans = embed(wav, segments, device, offset)
    long_i = [i for i, (s, e) in enumerate(spans) if e - s >= min_dur]
    if len(long_i) < n_speakers:
        raise ValueError(f"only {len(long_i)} segments over {min_dur}s, "
                         f"need at least {n_speakers} to cluster")

    # Cosine on L2-normalised vectors. n_clusters is fixed rather than
    # threshold-tuned: same-mic speakers sit closer together than a threshold
    # picked on any other recording would expect.
    fit = AgglomerativeClustering(n_clusters=n_speakers, metric="cosine",
                                  linkage="average").fit(x[long_i])
    labels = [None] * len(segments)
    for i, c in zip(long_i, fit.labels_):
        labels[i] = int(c)

    cen = {c: x[[i for i in long_i if labels[i] == c]].mean(axis=0)
           for c in set(fit.labels_)}
    cen = {c: v / np.linalg.norm(v) for c, v in cen.items()}

    # Short segments: trust their own embedding when it clearly favours one
    # centroid, else fall back to the nearest long segment in time. The
    # fallback fails where a short backchannel *starts* a turn, which is why
    # the margin exists at all.
    fallback = 0
    for i, (s, e) in enumerate(spans):
        if labels[i] is not None:
            continue
        sims = sorted(((float(x[i] @ v), c) for c, v in cen.items()),
                      reverse=True)
        if len(sims) > 1 and sims[0][0] - sims[1][0] >= margin:
            labels[i] = sims[0][1]
        else:
            labels[i] = labels[min(long_i, key=lambda k: abs(spans[k][0] - s))]
            fallback += 1

    # Order speakers by first appearance, so --names is given in speaking order.
    order, seen = {}, []
    for c in labels:
        if c not in order:
            order[c] = len(seen)
            seen.append(c)
    names = list(names or [])
    out = [names[order[c]] if order[c] < len(names) else f"S{order[c] + 1}"
           for c in labels]

    cents = np.vstack([cen[c] for c in seen if c in cen])
    stats = {
        "segments": len(segments),
        "short": len(segments) - len(long_i),
        "short_from_neighbour": fallback,
        "per_speaker": {n: out.count(n) for n in dict.fromkeys(out)},
        # The doc's warning made concrete: how far apart the voices sit.
        # Near 0 means the clustering split noise, not speakers.
        "centroid_cosine_distance": (float(1 - cents[0] @ cents[1])
                                     if len(cents) == 2 else None),
    }
    return out, stats
