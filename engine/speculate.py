"""Exact self-speculation for one sequence: propose from history, keep what greedy confirms.

Drafts are copied from the sequence's own tokens after the most recent earlier
occurrence of its current suffix (three tokens, else two, else one). The full
model then scores the trusted token and the drafts in one pass. A draft is kept
only if it equals the model's own greedy choice at that position, so the output
is the greedy sequence whatever the drafts were; bad drafts only cost time.
Everything here is fixed-shape tensor code, safe inside a CUDA graph.
"""

import torch


def propose(history, position, count, index):
    """``count`` draft tokens to follow ``history[position]``.

    ``history`` is int64 [C]; entries up to ``position`` (int64 [1]) are known,
    later ones may be stale. ``index`` is ``arange(C)``. A match at ``j`` must
    end before ``position`` so that its successor is a known token.
    """
    size = history.shape[0]
    last = history.index_select(0, position)
    before = history.index_select(0, (position - 1).clamp_min(0))
    earlier = history.index_select(0, (position - 2).clamp_min(0))
    one = (history == last) & (index < position)
    two = one & (torch.roll(history, 1) == before) & (index >= 1) & (position >= 1)
    three = two & (torch.roll(history, 2) == earlier) & (index >= 2) & (position >= 2)
    # Longer suffix first, then the most recent occurrence.
    rank = torch.where(three, index + 2 * size, torch.where(two, index + size, torch.where(one, index, index - size)))
    best = rank.max().reshape(1)
    start = torch.where(best >= 0, best % size, position)
    return history.index_select(0, (start + 1 + index[:count]).clamp_max(size - 1))


def accept(tokens, greedy):
    """Number of leading drafts the model itself chose, as int64 [1].

    ``tokens[0]`` is trusted and ``tokens[1:]`` are drafts; ``greedy[i]`` is the
    model's choice after ``tokens[:i + 1]``. Draft i+1 stands only if it equals
    ``greedy[i]`` and every earlier draft stood.
    """
    agree = (tokens[1:] == greedy[:-1]).to(torch.int64)
    return agree.cumprod(0).sum().reshape(1)
