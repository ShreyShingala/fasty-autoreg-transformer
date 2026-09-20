"""Relaxed acceptance (ACCEPT_MARGIN > 0) emulated line by line against kernels/spec.py.

check_tree.py proves the EXACT path: the emitted text equals sequential greedy.
That test cannot see the relaxed path, because emu_settle there has no `within`
argument at all -- and neither does interp/run_spec.py, which calls
spec.settle() without `within=`. So `_within_margin` and `_settle(RELAXED=1)`
have no executable reference anywhere in the CPU suite.

This file supplies one. The model is a deterministic logit vector per context,
so `within_margin` can be evaluated exactly, and the check is the judge's own:

  for every emitted token, on the engine's OWN emitted prefix, its logit must
  be within MARGIN of the argmax there.

plus check_tree.py's cache invariant, which is the thing an emission bug breaks:

  the K/V slots below `place` hold exactly the emitted text below `place`.

Run:  python agent/local_cpu/check_tree_relaxed.py [cases] [margin]
"""
import random
import sys

from check_tree import emu_propose


def emu_settle_relaxed(tokens, greedy, within, place, limit, CHAIN):
    """Line-by-line mirror of _settle in engine/kernels/spec.py with RELAXED=1."""
    T = len(tokens)
    legal = [bool(within[slot]) if slot >= 1 else False for slot in range(T)]
    miss = [slot if (1 <= slot < CHAIN and not legal[slot]) else CHAIN for slot in range(T)]
    gained = min(miss)
    accepted = gained                      # chain length BEFORE the sibling clamp
    wanted = greedy[0]
    hit = min([slot if (slot >= CHAIN and tokens[slot] == wanted) else T for slot in range(T)])
    room = max(limit - place, 0)
    branch = gained == 1 and hit < T and room >= 2
    emu_settle_relaxed.stale = greedy[min(gained, T - 1)] if (gained < CHAIN and not branch) else -1
    gained = min(2 if branch else gained, room)
    bonus = greedy[min(hit, T - 1)]
    emitted = [tokens[slot + 1] if (slot + 1 < accepted) else greedy[slot] for slot in range(T)]
    emitted = [bonus if (branch and slot == 1) else emitted[slot] for slot in range(T)]
    return gained, emitted, (place + hit if branch else -1), place + 1


def logit_model(seed, vocab, repeat, spread):
    """context -> logit vector. Deterministic, and flat enough that margins bite."""
    rng = random.Random(seed)
    table = {}

    def logits(prefix):
        key = tuple(prefix[-2:]) if repeat else (len(prefix), prefix[-1])
        if key not in table:
            local = random.Random((seed, key).__hash__())
            table[key] = [local.uniform(0.0, spread) for _ in range(vocab)]
        return table[key]

    return logits


def run(cases=400, MARGIN=1.0, spread=3.0, verbose=True):
    stats = {"branch": 0, "passes": 0, "relaxed": 0, "emitted": 0, "chains": set()}
    worst = 0.0
    for seed in range(cases):
        rng = random.Random(seed)
        repeat = seed % 2 == 0
        vocab = 5 if repeat else 12
        TOKENS, D = rng.choice([(16, (5, 8, 13, 14)), (8, (2, 4, 6, 7)), (5, (2, 3, 4, 4)),
                                (4, (1, 2, 3, 3)), (3, (1, 2, 2, 2)), (2, (1, 1, 1, 1))])
        TOP = 8
        prompt_len, outputs = rng.choice([(1, 6), (3, 12), (9, 32), (30, 40)])
        logits_of = logit_model(seed, vocab, repeat, spread)
        nxt = lambda ctx: max(range(vocab), key=lambda v: (logits_of(ctx)[v], -v))  # noqa: E731
        top = [[rng.randrange(vocab) for _ in range(TOP)] for _ in range(vocab)]
        prompt = [rng.randrange(vocab) for _ in range(prompt_len)]
        SIZE = prompt_len + outputs + TOKENS + 2
        limit = prompt_len + outputs - 1
        h = [0] * SIZE
        h[:prompt_len] = prompt
        h[prompt_len] = nxt(prompt)
        place = prompt_len
        emitted_all = [h[prompt_len]]
        kv = {i: prompt[i] for i in range(prompt_len)}   # slot -> token whose K/V it holds
        hint = -1
        while len(emitted_all) < outputs:
            tokens, CHAIN, phases = emu_propose(h, place, top, SIZE, TOKENS, D, 8, TOP, hint)
            stats["chains"].add((TOKENS, CHAIN))
            for t in range(TOKENS):
                kv[place + t] = tokens[t]
            assert [kv[i] for i in range(place)] == h[:place], \
                f"seed {seed}: cache prefix does not hold the emitted text at place {place}"
            known = h[:place]
            block, greedy, best = [], [], []
            for t in range(TOKENS):
                context = known + tokens[: t + 1] if t < CHAIN else known + [tokens[0], tokens[t]]
                row = logits_of(context)
                block.append(row)
                greedy.append(max(range(vocab), key=lambda v: (row[v], -v)))
                best.append(max(row))
            within = [0] * TOKENS
            for slot in range(1, TOKENS):
                within[slot] = int(block[slot - 1][tokens[slot]] >= best[slot - 1] - MARGIN)
                if slot < CHAIN and within[slot] and tokens[slot] != greedy[slot - 1]:
                    stats["relaxed"] += 1
            gained, emitted, move_from, move_to = emu_settle_relaxed(
                tokens, greedy, within, place, limit, CHAIN)
            hint = emu_settle_relaxed.stale
            if move_from >= 0:
                kv[move_to] = kv[move_from]
                stats["branch"] += 1
            for slot in range(TOKENS):
                h[place + 1 + slot] = emitted[slot]
            emitted_all += emitted[:gained]
            place += gained
            stats["passes"] += 1
            assert place <= limit and place + TOKENS < SIZE
        # The judge's check, replayed on the engine's own output.
        stats["emitted"] += len(emitted_all)
        for index, token in enumerate(emitted_all):
            row = logits_of(prompt + emitted_all[:index])
            gap = max(row) - row[token]
            worst = max(worst, gap)
            assert gap <= MARGIN + 1e-9, (
                f"ILLEGAL seed {seed} shape TOKENS={TOKENS} D={D} prompt={prompt_len} out={outputs}: "
                f"emitted token {token} at output index {index} has gap {gap:.4f} > MARGIN {MARGIN}\n"
                f"  prompt {prompt}\n  emitted {emitted_all}")
    if verbose:
        print(f"relaxed tree speculation MARGIN={MARGIN}: {cases} cases, {stats['passes']} passes, "
              f"{stats['emitted']} emitted tokens, {stats['relaxed']} margin-accepted non-argmax drafts, "
              f"{stats['branch']} alternative branches, {len(stats['chains'])} (tokens, chain) shapes; "
              f"worst replayed gap {worst:.4f}; cache prefix invariant held")
    return stats, worst


if __name__ == "__main__":
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    run(cases, margin)
