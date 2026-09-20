"""Run the RELAXED half of engine/kernels/spec.py in the Triton interpreter.

run_spec.py calls spec.settle() with no `within=`, so it only ever executes
_settle with RELAXED=0, and it never touches _within_margin at all. This script
covers what that leaves out:

  * _within_margin against real BF16 logits and the FP32 `top` that
    kernels/argmax.py writes, and
  * _settle with RELAXED=1, against check_tree_relaxed.emu_settle_relaxed.
"""
import contextlib
import io
import random
import sys

import interp_bf16  # noqa: F401  patches the interpreter's BF16 handling
import torch

with contextlib.redirect_stdout(io.StringIO()):
    import check_tree as emu                       # its self-test runs on import
    import check_tree_relaxed as relaxed

from kernels import spec
from kernels.argmax import argmax

rng = random.Random(7)
cases = 0
branches = 0
margin_accepts = 0
flags_checked = 0
for case in range(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
    TOKENS, D = rng.choice([(16, (5, 8, 13, 14)), (8, (2, 4, 6, 7)), (5, (2, 3, 4, 4)),
                            (4, (1, 2, 3, 3)), (2, (1, 1, 1, 1))])
    batch = rng.choice([1, 3, 4])
    vocab = rng.choice([4, 9, 40])
    SIZE = rng.choice([40, 97])
    TOP = 8
    MARGIN = rng.choice([0.25, 0.5, 1.0, 2.0])
    table = torch.tensor([[rng.randrange(vocab) for _ in range(TOP)] for _ in range(vocab)])
    history = torch.zeros(batch, SIZE, dtype=torch.int64)
    position = torch.zeros(batch, dtype=torch.int64)
    for b in range(batch):
        place = rng.randrange(0, SIZE - TOKENS - 2)
        position[b] = place
        history[b, :place + 1] = torch.tensor([rng.randrange(vocab) for _ in range(place + 1)])
    chains = torch.zeros(batch, dtype=torch.int64)
    phases = torch.zeros(batch, TOKENS, dtype=torch.int64)
    stale = torch.tensor([rng.choice([-1, -1, rng.randrange(vocab)]) for _ in range(batch)], dtype=torch.int64)
    tokens = spec.propose(history, position, TOKENS, D, table, stale, chains, phases)

    # Logits flat enough that the margin actually bites, as BF16 like the engine's.
    logits = (torch.randn(batch, TOKENS, vocab) * (MARGIN * 0.9)).bfloat16()
    top_logit = torch.zeros(batch * TOKENS, dtype=torch.float32)
    greedy = argmax(logits, top=top_logit).reshape(batch, TOKENS)
    within = spec.within_margin(logits.reshape(batch * TOKENS, vocab), tokens, top_logit, MARGIN)

    exact = logits.float()
    for b in range(batch):
        for slot in range(TOKENS):
            assert int(greedy[b, slot]) == int(exact[b, slot].argmax()), ("argmax", case, b, slot)
            assert abs(float(top_logit[b * TOKENS + slot]) - float(exact[b, slot].max())) < 1e-6, \
                ("top", case, b, slot)
        for slot in range(1, TOKENS):
            want = int(float(exact[b, slot - 1, int(tokens[b, slot])]) >= float(exact[b, slot - 1].max()) - MARGIN)
            assert int(within[b, slot]) == want, ("within_margin", case, b, slot, int(within[b, slot]), want)
            flags_checked += 1
            margin_accepts += want and int(tokens[b, slot]) != int(greedy[b, slot - 1])

    limit = position + torch.tensor([rng.randrange(0, TOKENS + 3) for _ in range(batch)])
    h2, p2 = history.clone(), position.clone()
    result = torch.zeros(batch, TOKENS + 1, dtype=torch.int64)
    move_from = torch.zeros(batch, dtype=torch.int64)
    move_to = torch.zeros(batch, dtype=torch.int64)
    spec.settle(tokens, greedy, p2, limit, h2, result, move_from, move_to, chains, stale, within=within)
    for b in range(batch):
        place = int(position[b])
        gained, emitted, mf, mt = relaxed.emu_settle_relaxed(
            tokens[b].tolist(), greedy[b].tolist(), within[b].tolist(), place, int(limit[b]), int(chains[b]))
        assert (int(result[b, 0]), result[b, 1:].tolist(), int(move_from[b]), int(move_to[b]), int(p2[b])) \
            == (gained, emitted, mf, mt, place + gained), \
            ("settle RELAXED", case, b, MARGIN, int(result[b, 0]), gained, result[b, 1:].tolist(), emitted)
        assert h2[b, place + 1: place + 1 + TOKENS].tolist() == emitted, ("history", case, b)
        assert int(stale[b]) == relaxed.emu_settle_relaxed.stale, ("stale", case, b)
        branches += mf >= 0
    cases += 1
print(f"relaxed spec.py in Triton interpreter == check_tree_relaxed emulation: {cases} cases "
      f"(_within_margin on {flags_checked} flags, {margin_accepts} of them non-argmax accepts; "
      f"_settle RELAXED=1 with {branches} alternative branches)")
