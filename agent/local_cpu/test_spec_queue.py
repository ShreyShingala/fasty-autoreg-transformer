"""CPU simulation of the host-side pass queue: exact count, order, bounded enqueueing."""
import random, sys
sys.path.insert(0, "/work/engine")
import torch
import decode
from decode import DecodeState, DRAFTS, SPEC_LOOKAHEAD

class Event:
    def record(self): pass
    def synchronize(self): pass

class Graph:
    def __init__(self, state, rng): self.state, self.rng, self.replays = state, rng, 0
    def replay(self):
        s = self.state; self.replays += 1
        gained = self.rng.randint(1, DRAFTS + 1)
        row = [gained] + [s.next_token + i for i in range(DRAFTS + 1)]
        s.next_token += gained
        s.result = torch.tensor(row)

for seed in range(300):
    rng = random.Random(seed)
    outputs = rng.choice([3, 4, 5, 8, 32, 33, 128])
    s = DecodeState.__new__(DecodeState)
    s.shape = (1, 7, outputs); s.speculative = True
    s.host_passes = torch.zeros((outputs, DRAFTS + 2), dtype=torch.int64)
    s.pass_events = [Event() for _ in range(outputs)]
    s.events = [Event()]; s.host_tokens = torch.tensor([[1000]])
    s.tokens, s.passes_enqueued, s.passes_read = [], 0, 0
    s.started, s.pace_seconds = 0.0, 0.0
    s.next_token = 1001
    s.spec_graph = Graph(s, rng)
    s.advance(5)
    got = [s.read(step) for step in range(outputs)]
    assert got == [[1000 + i] for i in range(outputs)], (seed, got[:6])
    assert s.passes_enqueued <= outputs - 1 and s.passes_enqueued - s.passes_read <= SPEC_LOOKAHEAD
    wasted = s.passes_enqueued - s.passes_read
    assert wasted <= SPEC_LOOKAHEAD
print("host pass queue: exact tokens in order for 300 random acceptance patterns; enqueueing bounded")
