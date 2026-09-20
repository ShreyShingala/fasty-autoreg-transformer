"""CPU check: batched speculation must reproduce sequential greedy for arbitrary deterministic 'models'."""
import random, sys
sys.path.insert(0, "/work/engine")
import torch
from speculate import propose, accept, advance

def make_model(seed, vocab, repeat):
    rng = random.Random(seed); table = {}
    def nxt(prefix):
        key = tuple(prefix[-2:]) if repeat else (len(prefix), prefix[-1])
        if key not in table: table[key] = rng.randrange(vocab)
        return table[key]
    return nxt

def run(seed, batch, prompt_len, outputs, k, vocab, repeat):
    rng = random.Random(seed * 7 + 1)
    models = [make_model(seed * 31 + b, vocab, repeat) for b in range(batch)]
    prompts = [[rng.randrange(vocab) for _ in range(prompt_len)] for _ in range(batch)]
    reference = []
    for b in range(batch):
        seq = list(prompts[b])
        for _ in range(outputs): seq.append(models[b](seq))
        reference.append(seq[prompt_len:])
    T = k + 1
    size = prompt_len + outputs + T + 1
    limit = prompt_len + outputs - 1
    history = torch.zeros((batch, size), dtype=torch.int64)
    history[:, :prompt_len] = torch.tensor(prompts)
    history[:, prompt_len] = torch.tensor([models[b](prompts[b]) for b in range(batch)])
    index = torch.arange(size); block = torch.arange(T)
    successor = torch.tensor([random.Random(seed * 13 + v).randrange(vocab) for v in range(vocab)])
    position = torch.full((batch,), prompt_len)
    emitted = [[int(history[b, prompt_len])] for b in range(batch)]
    passes = 0
    while min(len(e) for e in emitted) < outputs:
        drafts = propose(history, position, k, index, successor)
        assert drafts.shape == (batch, k)
        tokens = torch.cat((history.gather(1, position[:, None]), drafts), dim=1)
        greedy = torch.zeros((batch, T), dtype=torch.int64)
        for b in range(batch):
            known = history[b, : int(position[b])].tolist()
            for i in range(T): greedy[b, i] = models[b](known + tokens[b, : i + 1].tolist())
        positions = position[:, None] + block[None, :]
        assert int(positions.max()) + 1 < size and int(positions.max()) <= limit + T - 1
        gained, position = advance(position, accept(tokens, greedy) + 1, limit)
        history.scatter_(1, positions + 1, greedy)
        for b in range(batch): emitted[b] += greedy[b, : int(gained[b])].tolist()
        passes += 1
        assert passes <= outputs
    for b in range(batch):
        assert emitted[b] == reference[b], (seed, b, len(emitted[b]))
    return passes

total = 0
for seed in range(40):
    for repeat in (False, True):
        for batch, prompt_len, outputs, k in ((1, 1, 5, 4), (1, 7, 32, 4), (3, 2, 9, 3), (4, 33, 32, 3), (2, 64, 40, 6)):
            total += 1
            run(seed, batch, prompt_len, outputs, k, vocab=6 if repeat else 50, repeat=repeat)
print("batched speculation equals sequential greedy in", total, "cases")
