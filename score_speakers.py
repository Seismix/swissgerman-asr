"""Score out/<key>.speakers.txt against the speaker attribution in the docx.

The docx is edited prose and is NOT timestamp-aligned, so it cannot give a WER
(see data/README.md). It can give a *speaker* score: who said something is
objective in a way transcription convention is not. Segments are aligned to
docx turns by monotonic DP on content-word overlap, then the labels compared.
"""
import re, sys, pathlib

DOCX = pathlib.Path("data/transcripts/_docx_transkript.txt")
LABELLED = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "out/flix-ct2.speakers.txt")

def words(s):
    return {w for w in re.findall(r"[a-zäöüß]+", s.lower()) if len(w) >= 5}

turns = []
for line in DOCX.read_text(encoding="utf-8").splitlines():
    m = re.match(r"^(A|B):\s*(.*)", line)
    if m:
        turns.append((m.group(1), words(m.group(2))))

segs = []
for line in LABELLED.read_text(encoding="utf-8").splitlines():
    m = re.match(r"^\[([\d:]+)\]\s*(A|B):\s*(.*)", line)
    if m:
        segs.append((m.group(1), m.group(2), m.group(3)))

n, mlen = len(segs), len(turns)
sim = [[len(words(t) & tw) / max(1, len(words(t))) for _, tw in turns]
       for _, _, t in segs]

NEG = float("-inf")
dp = [[NEG] * mlen for _ in range(n)]
bk = [[-1] * mlen for _ in range(n)]
for j in range(mlen):
    dp[0][j] = sim[0][j]
for i in range(1, n):
    best, bestj = NEG, -1
    for j in range(mlen):
        if dp[i - 1][j] > best:
            best, bestj = dp[i - 1][j], j
        dp[i][j] = sim[i][j] + best
        bk[i][j] = bestj
j = max(range(mlen), key=lambda k: dp[n - 1][k])
path = [0] * n
for i in range(n - 1, -1, -1):
    path[i] = j
    j = bk[i][j]

# A segment the aligner could not place (no shared content words) tells us
# nothing about the labelling, so score those separately instead of counting
# them as diarization errors.
CONF = 0.30
ok = bad = conf_ok = conf_bad = unplaced = 0
rows = []
for i, (ts, pred, txt) in enumerate(segs):
    truth, s = turns[path[i]][0], sim[i][path[i]]
    hit = pred == truth
    ok, bad = ok + hit, bad + (not hit)
    if s < CONF:
        unplaced += 1
    else:
        conf_ok, conf_bad = conf_ok + hit, conf_bad + (not hit)
    if not hit:
        rows.append((ts, pred, truth, s, txt))

print(f"{LABELLED}  ({len(turns)} docx turns, {n} segments)")
print(f"  all segments        {ok}/{n} = {100 * ok / n:.1f}%")
d = conf_ok + conf_bad
print(f"  confidently aligned {conf_ok}/{d} = {100 * conf_ok / d:.1f}%  "
      f"(conf >= {CONF}; {unplaced} segments unplaceable)")
print("\nmismatches (ts, predicted, docx, align-confidence):")
for ts, pred, truth, s, txt in rows:
    flag = "  <- alignment unreliable" if s < CONF else ""
    print(f"  [{ts}] pred={pred:<6} docx={truth:<6} conf={s:.2f}  {txt[:70]}{flag}")
