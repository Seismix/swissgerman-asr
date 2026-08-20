"""Score out/<key>.speakers.txt against the speaker attribution in a reference.

    python score_speakers.py [labelled.txt] [reference.txt]

The reference is edited prose and is NOT timestamp-aligned, so it cannot give a
WER (see data/README.md). It can give a *speaker* score: who said something is
objective in a way transcription convention is not. Segments are aligned to
reference turns by monotonic DP on content-word overlap, then the labels
compared.

Speaker names are read from the labelled transcript - whatever `--names` put
there - and then used to parse the reference, so this works on any recording.
"""
import re, sys, pathlib

# Defaults to whatever run.py's default model writes. Pass a path for any other.
import asr
LABELLED = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else f"out/{asr.parse_model_spec()[1]}.speakers.txt")
DOCX = pathlib.Path(sys.argv[2] if len(sys.argv) > 2
                    else "data/transcripts/_docx_transkript.txt")

# "[00:01:23] Name: text" - the format transcript.as_txt writes with labels.
LABELLED_LINE = re.compile(r"^\[([\d:]+)\]\s*([^:]{1,40}?):\s*(.*)")
# "Name: text" - any attributed line, so we can tell a speaker we don't know
# about from prose that merely contains a colon.
ANY_TURN = re.compile(r"^([^:]{1,40}?):\s*(.*)")


def words(s):
    return {w for w in re.findall(r"[a-zäöüß]+", s.lower()) if len(w) >= 5}


if not LABELLED.exists():
    sys.exit(f"no such file: {LABELLED}  (run with --names to produce one)")
if not DOCX.exists():
    sys.exit(f"no such file: {DOCX}")

segs = []
for line in LABELLED.read_text(encoding="utf-8").splitlines():
    m = LABELLED_LINE.match(line)
    if m:
        segs.append((m.group(1), m.group(2), m.group(3)))
if not segs:
    sys.exit(f"{LABELLED} has no '[time] Name: text' lines - was it written "
             f"without --names?")

# The reference is parsed with the names the transcript actually uses, rather
# than a hardcoded pair. A name in the reference that no segment carries is
# reported instead of silently dropped: it would otherwise shrink the turn list
# and shift every alignment.
names = {pred for _, pred, _ in segs}
turns, unknown = [], {}
for line in DOCX.read_text(encoding="utf-8").splitlines():
    m = ANY_TURN.match(line)
    if not m:
        continue
    who, said = m.group(1).strip(), m.group(2)
    if who in names:
        turns.append((who, words(said)))
    else:
        unknown[who] = unknown.get(who, 0) + 1
if not turns:
    sys.exit(f"{DOCX} has no turns attributed to {sorted(names)} - check that "
             f"--names matches the names used in the reference")

n, mlen = len(segs), len(turns)
turn_words = [tw for _, tw in turns]
seg_words = [words(t) for _, _, t in segs]
# Kept alongside the ratio: the ratio alone cannot distinguish "8 of 10 content
# words match this turn" from "the segment's single content word happens to
# appear in it", and those are not equally good evidence.
shared = [[len(sw & tw) for tw in turn_words] for sw in seg_words]
sim = [[sh / max(1, len(sw)) for sh in row]
       for sw, row in zip(seg_words, shared)]

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
# A second, stricter tier, reported but NOT the headline. It was chosen after
# seeing the result, so quoting it as an accuracy would be tuning; it is here
# because it says something the ratio cannot - see the note printed below.
MIN_SHARED = 2
ok = bad = conf_ok = conf_bad = unplaced = 0
strict_ok = strict_n = thin = 0
rows = []
for i, (ts, pred, txt) in enumerate(segs):
    truth, s = turns[path[i]][0], sim[i][path[i]]
    sh = shared[i][path[i]]
    hit = pred == truth
    ok, bad = ok + hit, bad + (not hit)
    if s < CONF:
        unplaced += 1
    else:
        conf_ok, conf_bad = conf_ok + hit, conf_bad + (not hit)
        if sh >= MIN_SHARED:
            strict_n, strict_ok = strict_n + 1, strict_ok + hit
        else:
            thin += 1
    if not hit:
        rows.append((ts, pred, truth, s, sh, txt))

print(f"{LABELLED}  ({len(turns)} reference turns, {n} segments, "
      f"speakers: {', '.join(sorted(names))})")
for who, count in sorted(unknown.items(), key=lambda kv: -kv[1]):
    print(f"  note: {count} reference line(s) attributed to {who!r}, which no "
          f"segment carries - not scored")
print(f"  all segments        {ok}/{n} = {100 * ok / n:.1f}%")
d = conf_ok + conf_bad
if d:
    print(f"  confidently aligned {conf_ok}/{d} = {100 * conf_ok / d:.1f}%  "
          f"(conf >= {CONF}; {unplaced} segments unplaceable)")
else:
    print(f"  confidently aligned  n/a - all {n} segments below conf {CONF}")
if strict_n:
    print(f"  ...of which {thin} rest on a single shared content word; drop "
          f"those and it is {strict_ok}/{strict_n} = "
          f"{100 * strict_ok / strict_n:.1f}%")
    print(f"     (diagnostic, not a score: the >= {MIN_SHARED}-word floor was "
          f"chosen after seeing the result)")

print("\nmismatches (ts, predicted, reference, align-confidence, shared words):")
for ts, pred, truth, s, sh, txt in rows:
    flag = "  <- alignment unreliable" if s < CONF else (
        "  <- rests on 1 shared word" if sh < MIN_SHARED else "")
    print(f"  [{ts}] pred={pred:<6} ref={truth:<6} conf={s:.2f} n={sh:<2} "
          f"{txt[:64]}{flag}")
