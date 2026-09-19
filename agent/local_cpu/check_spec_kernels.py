"""Emulate kernels/spec.py formulas line by line and compare with the validated tensor reference."""
import random, sys
sys.path.insert(0, "/work/engine")
import torch
from speculate import propose, accept, advance

def emu_propose(history, position, successor, count):
    B, SIZE = history.shape; out = torch.zeros(B, count + 1, dtype=torch.int64)
    for row in range(B):
        h = history[row].tolist(); place = int(position[row])
        last, before, earlier = h[place], h[max(place - 1, 0)], h[max(place - 2, 0)]
        best = -1
        for index in range(SIZE):
            back1 = h[index - 1] if index >= 1 else -1; back2 = h[index - 2] if index >= 2 else -1
            two = index < place and h[index] == last and index >= 1 and back1 == before and place >= 1
            three = two and index >= 2 and back2 == earlier and place >= 2
            rank = index + SIZE if three else index if two else -1
            best = max(best, rank)
        found = best >= 0; start = best % SIZE if found else place
        out[row, 0] = last; previous = last
        for step in range(1, count + 1):
            source = start + step
            copied = h[min(source, SIZE - 1)]; followed = int(successor[previous])
            draft = copied if (found and source <= place) else followed
            out[row, step] = draft; previous = draft
    return out

def emu_settle(tokens, greedy, position, limit, history, result):
    B, T = tokens.shape
    for row in range(B):
        miss = [slot if (slot >= 1 and int(tokens[row, slot]) != int(greedy[row, slot - 1])) else T for slot in range(T)]
        gained = min(miss); place = int(position[row])
        gained = min(gained, max(int(limit[row]) - place, 0))
        result[row, 0] = gained; result[row, 1:] = greedy[row]
        history[row, place + 1: place + 1 + T] = greedy[row]
        position[row] = place + gained

rng = random.Random(0)
for case in range(400):
    B, V, count = rng.choice([1, 2, 4]), rng.choice([4, 9, 60]), rng.choice([1, 3, 4])
    SIZE = rng.choice([12, 40, 97]); T = count + 1
    history = torch.tensor([[rng.randrange(V) for _ in range(SIZE)] for _ in range(B)])
    position = torch.tensor([rng.randrange(0, SIZE - T - 1) for _ in range(B)])
    for b in range(B): history[b, int(position[b]) + 1:] = 0
    successor = torch.tensor([rng.randrange(V) for _ in range(V)])
    want = torch.cat((history.gather(1, position[:, None]), propose(history, position, count, torch.arange(SIZE), successor)), 1)
    got = emu_propose(history, position, successor, count)
    assert torch.equal(got, want), (case, got, want)
    greedy = torch.tensor([[rng.randrange(V) for _ in range(T)] for _ in range(B)])
    for b in range(B):                      # make some drafts agree
        for i in range(1, T):
            if rng.random() < 0.6: greedy[b, i - 1] = got[b, i]
    limit = position + torch.tensor([rng.randrange(0, T + 3) for _ in range(B)])
    g_ref, p_ref = advance(position, accept(got, greedy) + 1, limit)
    h2, pos2, res = history.clone(), position.clone(), torch.zeros(B, T + 1, dtype=torch.int64)
    emu_settle(got, greedy, pos2, limit, h2, res)
    assert torch.equal(res[:, 0], g_ref) and torch.equal(pos2, p_ref) and torch.equal(res[:, 1:], greedy)
    href = history.clone(); href.scatter_(1, position[:, None] + torch.arange(T)[None, :] + 1, greedy)
    assert torch.equal(h2, href)
print("fused propose/settle formulas equal the tensor reference in 400 random cases")
