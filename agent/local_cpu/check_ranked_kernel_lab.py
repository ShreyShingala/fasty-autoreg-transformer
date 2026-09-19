"""Line-by-line emulation of engine/kernels/spec.py::_propose_ranked, compared with the lab reference r1_block (R1, no skip-gram)."""
import json, math, sys, collections, torch
sys.argv = ["x", "r1"]
import r1_vector as ref                      # loads weights R1 and the table used by the lab
from dl import samples, SHIPPED
W = json.load(open("cache/final_weights_r1.json"))["all"]
weights = [W[f"best{k}"] for k in range(9)] + [W[f"tr{k}"] for k in range(8)] + [W["trnone"], W["first"], W["cnt_all"] / 3, W["uni"] / 5]
top8 = torch.from_numpy(ref.TOP()).tolist()
MAPS = {16: (5, 8, 13, 14), 8: (2, 4, 6, 7), 5: (2, 3, 4, 4), 4: (1, 2, 3, 3), 3: (1, 2, 2, 2), 2: (1, 1, 1, 1)}
assert all(tuple(SHIPPED[T]) == MAPS[T] for T in MAPS), "lab maps differ from the engine's"

def emu(h, place, TOKENS, D, SIZE, MAXLEN=8, HIST=8, TOP=8, BLOCK_C=16):
    last = h[place]; NEG = float("-inf")
    one = [i < place and h[i] == last for i in range(SIZE)]
    length = [1 if o else 0 for o in one]; agree = list(one)
    for back in range(1, MAXLEN):
        wanted = h[max(place - back, 0)]
        for i in range(SIZE):
            seen = h[i - back] if i >= back else -1
            agree[i] = agree[i] and i >= back and place >= back and seen == wanted
            length[i] += 1 if agree[i] else 0
    rank = [i + (length[i] - 1) * SIZE if one[i] else -1 for i in range(SIZE)]
    after = [h[i + 1] if one[i] else -1 for i in range(SIZE)]
    cand = [-1] * BLOCK_C; count = 0; open_rank = list(rank)
    for _ in range(HIST):
        choice = max(open_rank); usable = choice >= 0
        token = h[max(choice, 0) % SIZE + 1]
        if usable: cand[count] = token
        open_rank = [-1 if (one[i] and after[i] == token) else open_rank[i] for i in range(SIZE)]
        count += 1 if usable else 0
    table_rank = [TOP] * BLOCK_C
    for entry in range(TOP):
        token = top8[last][entry]
        known = [cand[l] == token and l < count for l in range(BLOCK_C)]
        table_rank = [entry if known[l] else table_rank[l] for l in range(BLOCK_C)]
        add = (sum(known) == 0) and count < BLOCK_C
        if add: cand[count] = token; table_rank[count] = entry
        count += 1 if add else 0
    score = []; suffix = []; origin = []
    for l in range(BLOCK_C):
        f = [one[i] and after[i] == cand[l] for i in range(SIZE)]
        followed = sum(f); br = max([rank[i] if f[i] else -1 for i in range(SIZE)])
        sfx = br // SIZE + 1 if br >= 0 else 0; org = br % SIZE if br >= 0 else 0
        occurs = sum(1 for i in range(SIZE) if i <= place and h[i] == cand[l])
        s = weights[sfx] + weights[9 + table_rank[l]] + (weights[18] if (l == 0 and sfx > 0) else 0.0)
        s += weights[19] * math.log(1.0 + followed) + weights[20] * math.log(1.0 + occurs)
        score.append(s if l < count else NEG); suffix.append(sfx); origin.append(org)
    top = max(range(BLOCK_C), key=lambda l: (score[l], -l)); first = cand[top]; matched = suffix[top]; start = origin[top]; score[top] = NEG
    drafts = D[0] if matched <= 1 else D[1] if matched <= 3 else D[2] if matched <= 7 else D[3]
    out = [None] * TOKENS; out[0] = last; out[1] = first; previous = first
    for step in range(2, TOKENS):
        source = start + step; copied = h[min(source, SIZE - 1)]; follow = top8[previous][0]
        draft = copied if (matched > 0 and source <= place) else follow
        if drafts >= step: out[step] = draft
        previous = draft
    for lane in range(TOKENS - 2):
        pick = max(range(BLOCK_C), key=lambda l: (score[l], -l)); value = score[pick]
        token = cand[pick] if value > NEG else first
        if 1 + drafts + lane < TOKENS: out[1 + drafts + lane] = token
        score[pick] = NEG
    assert None not in out
    return out, drafts

bad = tot = 0
for key in ("p512", "p2048"):
    for s in samples(key)[::13]:
        seq = s["prompt"] + s["output"]; Lp = len(s["prompt"])
        for t in range(1, len(s["output"]), 7):
            hist = seq[:Lp + t]; place = len(hist) - 1; SIZE = len(seq) + 20; h = hist + [0] * (SIZE - len(hist))
            for T in (16, 8, 4, 2):
                chain, alts = ref.r1_block(torch.tensor(hist), T, torch.tensor(top8))
                out, drafts = emu(h, place, T, MAPS[T], SIZE)
                tot += 1
                if out[1:1 + drafts] != chain or out[1 + drafts:] != alts + [out[1]] * (T - 1 - drafts - len(alts)):
                    bad += 1
                    if bad <= 3: print("MISMATCH", key, t, T, out, chain, alts)
print("kernel emulation vs lab reference:", tot, "blocks,", bad, "mismatches")
