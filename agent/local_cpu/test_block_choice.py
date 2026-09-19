"""Block-size candidates per batch and the pass model that faces plain decode (pure host arithmetic).

Needs the engine's imports to resolve without Triton: run it where smoke_engine.py runs,
    ~/.cache/fasty-lab/venv/bin/python agent/local_cpu/test_block_choice.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "shim"), os.path.join(os.path.dirname(os.path.dirname(HERE)), "engine")]
import torch  # noqa: E402,F401
import torch._dynamo  # noqa: E402,F401  before the fake triton, as smoke_engine.py does
import decode  # noqa: E402

# Batches 1-16 are what the platform has scored: unchanged.
assert [decode.block_candidates(b) for b in (1, 2, 4, 8, 9, 16)] == [[16], [8, 16], [4, 8], [2, 4], [3, 5], [2, 4]]
for batch in range(17, 80):
    sizes = decode.block_candidates(batch)
    if batch > decode.MAX_SPECULATIVE_BATCH:
        assert sizes == [], batch
        continue
    assert len(sizes) == 1 and sizes[0] in (2, 3), (batch, sizes)
    size = sizes[0]
    assert size * batch <= 128 and (size == 2 or size * batch <= 64), (batch, size)
    assert size in decode.DRAFTS_BY_MATCH and size in decode.BATCH_PASSES
    for output in (8, 32, 128, 1024):
        passes = decode.expected_passes(size, batch, output)
        # Never better than one row alone would do offline, never worse than plain decode.
        assert 0.80 < passes <= 1.0, (batch, output, passes)
        assert passes >= decode.expected_passes(size, 17, output)
    assert decode.expected_passes(size, batch, 128) < decode.expected_passes(size, batch, 32)
print("block candidates and pass model: ok")
