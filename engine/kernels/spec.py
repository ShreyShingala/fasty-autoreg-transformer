"""Fused bookkeeping for exact speculation: one launch to propose, one to settle.

These are the tensor programs of ``speculate.propose`` and
``speculate.accept``/``advance`` (which remain the readable reference and are
what the CPU tests exercise), written as one Triton program per row so that a
verify pass does not pay for some sixty tiny elementwise launches.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _propose(
    history, position, successor, tokens,
    SIZE: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = history + row * SIZE
    place = tl.load(position + row)
    last = tl.load(base + place)
    before = tl.load(base + tl.maximum(place - 1, 0))
    earlier = tl.load(base + tl.maximum(place - 2, 0))
    index = tl.arange(0, BLOCK).to(tl.int64)
    inside = index < SIZE
    here = tl.load(base + index, inside, other=-1)
    back1 = tl.load(base + index - 1, inside & (index >= 1), other=-1)
    back2 = tl.load(base + index - 2, inside & (index >= 2), other=-1)
    # A suffix of at least two tokens ending before the position.
    two = inside & (index < place) & (here == last) & (index >= 1) & (back1 == before) & (place >= 1)
    three = two & (index >= 2) & (back2 == earlier) & (place >= 2)
    # Longer suffix first, then the most recent occurrence.
    rank = tl.where(three, index + SIZE, tl.where(two, index, -1))
    best = tl.max(rank, axis=0)
    found = best >= 0
    start = tl.where(found, best % SIZE, place)
    out = tokens + row * (COUNT + 1)
    tl.store(out, last)
    previous = last
    for step in tl.static_range(1, COUNT + 1):
        source = start + step
        copied = tl.load(base + tl.minimum(source, SIZE - 1))
        followed = tl.load(successor + previous)
        draft = tl.where(found & (source <= place), copied, followed)
        tl.store(out + step, draft)
        previous = draft


def propose(history, position, count, successor):
    """Trusted token plus ``count`` drafts per row, int64 [B, count + 1]."""
    batch, size = history.shape
    assert history.is_contiguous() and history.dtype == position.dtype == successor.dtype == torch.int64
    assert position.shape == (batch,) and successor.is_contiguous()
    tokens = torch.empty((batch, count + 1), dtype=torch.int64, device=history.device)
    _propose[(batch,)](
        history, position, successor, tokens,
        SIZE=size, COUNT=count, BLOCK=triton.next_power_of_2(size), num_warps=4,
    )
    return tokens


@triton.jit
def _settle(
    tokens, greedy, position, limit, history, result,
    SIZE: tl.constexpr, TOKENS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.arange(0, BLOCK).to(tl.int64)
    inside = slot < TOKENS
    draft = tl.load(tokens + row * TOKENS + slot, inside, other=0)
    chosen = tl.load(greedy + row * TOKENS + slot, inside, other=0)
    expected = tl.load(greedy + row * TOKENS + slot - 1, inside & (slot >= 1), other=0)
    # Draft ``slot`` stands only if it equals the model's choice after the
    # tokens before it; the first miss ends the run. Gained = drafts kept + 1.
    miss = tl.where(inside & (slot >= 1) & (draft != expected), slot, TOKENS)
    gained = tl.min(miss, axis=0)
    place = tl.load(position + row)
    gained = tl.minimum(gained, tl.maximum(tl.load(limit + row) - place, 0))
    tl.store(result + row * (TOKENS + 1), gained)
    tl.store(result + row * (TOKENS + 1) + 1 + slot, chosen, inside)
    # chosen[i] follows tokens[:i + 1]; entries past ``gained`` are rewritten
    # by the next pass before anything reads them.
    tl.store(history + row * SIZE + place + 1 + slot, chosen, inside)
    tl.store(position + row, place + gained)


def settle(tokens, greedy, position, limit, history, result):
    """Accept, clamp at ``limit``, record greedy tokens and move each row, in place."""
    batch, count = tokens.shape
    assert tokens.is_contiguous() and greedy.is_contiguous() and history.is_contiguous() and result.is_contiguous()
    assert greedy.shape == tokens.shape and result.shape == (batch, count + 1)
    assert tokens.dtype == greedy.dtype == position.dtype == limit.dtype == history.dtype == result.dtype == torch.int64
    _settle[(batch,)](
        tokens, greedy, position, limit, history, result,
        SIZE=history.shape[1], TOKENS=count, BLOCK=triton.next_power_of_2(count), num_warps=1,
    )
