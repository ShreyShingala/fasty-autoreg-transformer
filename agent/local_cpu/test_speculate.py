"""CPU check: speculation must reproduce sequential greedy for arbitrary deterministic 'models'."""
import random, sys
sys.path.insert(0, "/work/engine")
import torch
from speculate import propose, accept

def make_model(seed, vocab, repeat):
    rng = random.Random(seed)
    table = {}
    def nxt(prefix):
        # depends on the last two tokens; 'repeat' makes continuations copy earlier text
        key = tuple(prefix[-2:]) if repeat else (len(prefix), prefix[-1])
        if key not in table: table[key] = rng.randrange(vocab)
        return table[key]
    return nxt

def run(seed, prompt_len, outputs, k, vocab, repeat):
    rng = random.Random(seed * 7 + 1)
    nxt = make_model(seed, vocab, repeat)
    prompt = [rng.randrange(vocab) for _ in range(prompt_len)]
    reference = list(prompt)
    for _ in range(outputs): reference.append(nxt(reference))
    T = k + 1
    size = prompt_len + outputs + 3 * T + 2
    history = torch.randint(0, vocab, (size,), dtype=torch.int64)      # stale junk everywhere
    history[:prompt_len] = torch.tensor(prompt)
    history[prompt_len] = nxt(prompt)                                   # the prefill token
    index = torch.arange(size)
    block = torch.arange(T)
    position = torch.tensor([prompt_len])
    emitted, passes = [int(history[prompt_len])], 0
    while len(emitted) < outputs:
        drafts = propose(history, position, k, index)
        tokens = torch.cat((history.index_select(0, position), drafts))
        p = int(position)
        known = history[:p].tolist()
        greedy = torch.tensor([nxt(known + tokens[: i + 1].tolist()) for i in range(T)])
        count = accept(tokens, greedy) + 1
        history.index_copy_(0, position + block + 1, greedy)
        emitted += greedy[: int(count)].tolist()
        position = position + count
        passes += 1
        assert int(position) + T + 1 < size
    assert emitted[:outputs] == reference[prompt_len:prompt_len + outputs], (seed, emitted[:8], reference[prompt_len:prompt_len + 8])
    return passes

total = 0
for seed in range(60):
    for repeat in (False, True):
        for prompt_len, outputs, k in ((1, 5, 4), (2, 9, 4), (7, 32, 4), (64, 40, 3), (33, 64, 6)):
            total += 1
            passes = run(seed, prompt_len, outputs, k, vocab=6 if repeat else 50, repeat=repeat)
print("speculation equals sequential greedy in", total, "cases")
print("example: repetitive text, 64 outputs, k=6 ->", run(3, 33, 64, 6, 6, True), "passes")
