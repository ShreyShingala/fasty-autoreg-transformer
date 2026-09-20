"""CPU simulation of the host-side pass queue: exact count, order, bounded enqueueing."""
import random, sys
sys.path.insert(0, "/work/engine")
import torch
import decode
from decode import DecodeState, SPEC_LOOKAHEAD
DRAFTS = 4

class Event:
    """Completes only when synchronized, or (randomly) when queried: exercises both paths."""
    def __init__(self, rng): self.rng, self.done = rng, False
    def record(self): self.done = False
    def synchronize(self): self.done = True
    def query(self):
        self.done = self.done or self.rng.random() < 0.5
        return self.done

class Graph:
    def __init__(self, state, rng): self.state, self.rng, self.replays = state, rng, 0
    def replay(self):
        s = self.state; self.replays += 1
        rows = []
        for b in range(s.shape[0]):
            room = s.shape[2] - s.produced[b]          # the GPU clamps at the last requested token
            gained = min(self.rng.randint(1, DRAFTS + 1), room)
            rows.append([gained] + [1000 * (b + 1) + s.produced[b] + i for i in range(DRAFTS + 1)])
            s.produced[b] += gained
        s.result = torch.tensor(rows)

for seed in range(300):
    rng = random.Random(seed)
    outputs = rng.choice([3, 4, 5, 8, 32, 33, 128])
    batch = rng.choice([1, 1, 2, 4])
    s = DecodeState.__new__(DecodeState)
    s.shape = (batch, 7, outputs); s.speculative = True
    s.host_passes = torch.zeros((outputs, batch, DRAFTS + 2), dtype=torch.int64)
    s.pass_events = [Event(rng) for _ in range(outputs)]
    s.events = [Event(rng)]; s.host_tokens = torch.tensor([[1000 * (b + 1) for b in range(batch)]])
    s.tokens, s.passes_enqueued, s.passes_read = [], 0, 0
    s.started, s.pace_seconds, s.pass_seconds = 0.0, 0.0, 0.0
    s.natural, s.finished = [], None
    s.produced = [1] * batch
    s.spec_graph = Graph(s, rng)
    s.advance(5)
    got = [s.read(step) for step in range(outputs)]
    assert got == [[1000 * (b + 1) + i for b in range(batch)] for i in range(outputs)], (seed, got[:4])
    assert s.passes_enqueued <= outputs - 1 and s.passes_enqueued - s.passes_read <= SPEC_LOOKAHEAD
    wasted = s.passes_enqueued - s.passes_read
    assert wasted <= SPEC_LOOKAHEAD and s.finished is not None
print("host pass queue: exact per-row tokens in order for 300 random patterns (batched, query/synchronize paths); enqueueing bounded")
