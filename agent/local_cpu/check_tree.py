"""Emulate kernels/spec.py (tree propose + settle) line by line; the emitted text must equal sequential greedy."""
import random

def emu_propose(h, place, top, SIZE, DRAFTS, SIBLINGS, ALTERNATES, TOP):
    last, before, earlier = h[place], h[max(place - 1, 0)], h[max(place - 2, 0)]
    rank = []; one_mask = []
    for index in range(SIZE):
        back1 = h[index - 1] if index >= 1 else -1; back2 = h[index - 2] if index >= 2 else -1
        one = index < place and h[index] == last
        two = one and index >= 1 and back1 == before and place >= 1
        three = two and index >= 2 and back2 == earlier and place >= 2
        rank.append(index + 2 * SIZE if three else index + SIZE if two else index if one else -1); one_mask.append(one)
    best = max(rank); found = best >= 0; start = best % SIZE if found else place
    out = [last]; previous = last; first = last
    for step in range(1, DRAFTS + 1):
        source = start + step
        copied = h[min(source, SIZE - 1)]; followed = top[previous][0]
        draft = copied if (found and source <= place) else followed
        out.append(draft); previous = draft
        if step == 1: first = draft
    if SIBLINGS > 0:
        siblings = [-1] * SIBLINGS; count = 0
        after = [h[i + 1] if one_mask[i] else -1 for i in range(SIZE)]
        for _ in range(ALTERNATES):
            choice = max((rank[i] if (one_mask[i] and not (after[i] == first or after[i] in siblings)) else -1) for i in range(SIZE))
            usable = choice >= 0 and count < SIBLINGS
            candidate = h[max(choice, 0) % SIZE + 1]
            if usable: siblings[count] = candidate; count += 1
        for entry in range(TOP):
            candidate = top[last][entry]
            usable = candidate != first and candidate not in siblings and count < SIBLINGS
            if usable: siblings[count] = candidate; count += 1
        out += [s if s >= 0 else first for s in siblings]
    return out

def emu_settle(tokens, greedy, place, limit, CHAIN):
    T = len(tokens)
    miss = [slot if (1 <= slot < CHAIN and tokens[slot] != greedy[slot - 1]) else CHAIN for slot in range(T)]
    gained = min(miss); wanted = greedy[0]
    hit = min([slot if (slot >= CHAIN and tokens[slot] == wanted) else T for slot in range(T)])
    room = max(limit - place, 0)
    branch = gained == 1 and hit < T and room >= 2
    gained = min(2 if branch else gained, room)
    bonus = greedy[min(hit, T - 1)]
    emitted = [bonus if (branch and slot == 1) else greedy[slot] for slot in range(T)]
    return gained, emitted, (place + hit if branch else -1), place + 1

def make_model(seed, vocab, repeat):
    rng = random.Random(seed); table = {}
    def nxt(prefix):
        key = tuple(prefix[-2:]) if repeat else (len(prefix), prefix[-1])
        if key not in table: table[key] = rng.randrange(vocab)
        return table[key]
    return nxt

stats = {"branch": 0, "passes": 0}
for seed in range(300):
    rng = random.Random(seed); repeat = seed % 2 == 0; vocab = 5 if repeat else 12
    DRAFTS, SIBLINGS = rng.choice([(1, 0), (2, 1), (4, 3), (8, 7), (1, 3)]); TOP = 8; CHAIN = 1 + DRAFTS; T = CHAIN + SIBLINGS
    prompt_len, outputs = rng.choice([(1, 6), (3, 12), (9, 32), (30, 40)])
    nxt = make_model(seed, vocab, repeat)
    top = [[rng.randrange(vocab) for _ in range(TOP)] for _ in range(vocab)]
    prompt = [rng.randrange(vocab) for _ in range(prompt_len)]
    ref = list(prompt)
    for _ in range(outputs): ref.append(nxt(ref))
    SIZE = prompt_len + outputs + T + 2; limit = prompt_len + outputs - 1
    h = [0] * SIZE; h[:prompt_len] = prompt; h[prompt_len] = nxt(prompt); place = prompt_len
    emitted_all = [h[prompt_len]]
    # a model of the KV cache: slot -> token whose K/V it holds (what attention would actually read)
    kv = {i: prompt[i] for i in range(prompt_len)}
    while len(emitted_all) < outputs:
        tokens = emu_propose(h, place, top, SIZE, DRAFTS, SIBLINGS, min(SIBLINGS, 3), TOP)
        assert len(tokens) == T and tokens[0] == h[place]
        for t in range(T): kv[place + t] = tokens[t]                      # the fused kernel writes every block slot
        assert [kv[i] for i in range(place)] == h[:place], "cache prefix must hold exactly the known text"
        known = h[:place]; greedy = []
        for t in range(T):
            context = known + tokens[: t + 1] if t < CHAIN else known + [tokens[0], tokens[t]]   # tree mask
            greedy.append(nxt(context))
        gained, emitted, move_from, move_to = emu_settle(tokens, greedy, place, limit, CHAIN)
        if move_from >= 0: kv[move_to] = kv[move_from]; stats["branch"] += 1
        for slot in range(T): h[place + 1 + slot] = emitted[slot]
        emitted_all += emitted[:gained]; place += gained; stats["passes"] += 1
        assert place <= limit and place + T < SIZE
    assert emitted_all == ref[prompt_len:], (seed, emitted_all[:10], ref[prompt_len:prompt_len + 10])
    assert len(emitted_all) == outputs
print(f"tree speculation equals sequential greedy in 300 cases ({stats['branch']} sibling branches in {stats['passes']} passes); cache prefix invariant held")
