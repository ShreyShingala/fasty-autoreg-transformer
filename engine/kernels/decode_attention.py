"""Dense GQA decode with a device-side length and split-KV softmax reduction.

BF16 Q/K/V, FP32 scores and accumulation, BF16 output. Each partial attends to
a disjoint interval; the second kernel combines their softmax normalizers.
Every valid cache position contributes. No per-token host synchronization.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_partials(
    q_ptr, k_ptr, v_ptr, position_ptr, partial_ptr, stats_ptr,
    GROUPS: tl.constexpr, DIM: tl.constexpr, CAPACITY: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)  # flattened (batch, KV head)
    split = tl.program_id(1)
    heads = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, DIM)
    columns = tl.arange(0, BLOCK_N)
    query = tl.load(
        q_ptr + (group * GROUPS + heads[:, None]) * DIM + dims[None, :],
        heads[:, None] < GROUPS, other=0,
    )
    valid = tl.load(position_ptr).to(tl.int32) + 1
    begin = split * CHUNK
    end = tl.minimum(tl.minimum(begin + CHUNK, CAPACITY), valid)
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, DIM), tl.float32)
    cache_base = group * CAPACITY * DIM
    for start in range(begin, end, BLOCK_N):
        tokens = start + columns
        key = tl.load(
            k_ptr + cache_base + tokens[None, :] * DIM + dims[:, None],
            tokens[None, :] < end, other=0,
        )
        scores = tl.dot(query, key) * (SCALE * 1.4426950408889634)
        scores = tl.where(tokens[None, :] < end, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        probabilities = tl.exp2(scores - next_maximum[:, None])
        correction = tl.exp2(maximum - next_maximum)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        value = tl.load(
            v_ptr + cache_base + tokens[:, None] * DIM + dims[None, :],
            tokens[:, None] < end, other=0,
        )
        # Standard BF16 flash-attention product with FP32 accumulation.
        accumulator = tl.dot(probabilities.to(tl.bfloat16), value, accumulator)
        maximum = next_maximum
    partial_base = ((group * SPLITS + split) * GROUPS + heads) * DIM
    tl.store(
        partial_ptr + partial_base[:, None] + dims[None, :], accumulator,
        heads[:, None] < GROUPS,
    )
    stats_base = ((group * SPLITS + split) * GROUPS + heads) * 2
    # An empty interval has m=-inf, l=0, accumulator=0. The merge gives it
    # exactly zero weight, including when valid length is shorter than CHUNK.
    tl.store(stats_ptr + stats_base, maximum, heads < GROUPS)
    tl.store(stats_ptr + stats_base + 1, denominator, heads < GROUPS)


@triton.jit
def _decode_merge(
    partial_ptr, stats_ptr, out_ptr,
    GROUPS: tl.constexpr, DIM: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    head = tl.program_id(0).to(tl.int64)
    group = head // GROUPS
    within_group = head % GROUPS
    splits = tl.arange(0, BLOCK_S)
    dims = tl.arange(0, DIM)
    offset = (group * SPLITS + splits) * GROUPS + within_group
    maxima = tl.load(stats_ptr + offset * 2, splits < SPLITS, other=-float("inf"))
    denominators = tl.load(stats_ptr + offset * 2 + 1, splits < SPLITS, other=0)
    maximum = tl.max(maxima, axis=0)
    correction = tl.exp2(maxima - maximum)
    partials = tl.load(
        partial_ptr + offset[:, None] * DIM + dims[None, :],
        splits[:, None] < SPLITS, other=0,
    )
    denominator = tl.sum(denominators * correction, axis=0)
    numerator = tl.sum(partials * correction[:, None], axis=0)
    tl.store(out_ptr + head * DIM + dims, numerator / denominator)


def decode_attention(query, key, value, position, scale):
    """Q [B,Hq,1,D], KV [B,Hkv,C,D] -> [B,1,Hq,D]."""
    batch, query_heads, tokens, dim = query.shape
    kv_heads, capacity = key.shape[1:3]
    assert tokens == 1 and query_heads % kv_heads == 0
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    assert value.shape == key.shape and key.shape[0] == batch and key.shape[3] == dim
    assert dim in (64, 128) and position.shape == (1,) and position.dtype == torch.int64
    groups = query_heads // kv_heads
    block_n = 32 if batch * kv_heads < 16 else 64
    # Enough independent KV intervals for small batches to occupy an H100.
    # Shape-only policy: neither token values nor sample number affect it.
    splits = min(32, triton.cdiv(256, batch * kv_heads), triton.cdiv(capacity, block_n))
    chunk = triton.cdiv(capacity, splits)
    partial = torch.empty((batch * kv_heads, splits, groups, dim), device=query.device, dtype=torch.float32)
    stats = torch.empty((batch * kv_heads, splits, groups, 2), device=query.device, dtype=torch.float32)
    out = torch.empty((batch, 1, query_heads, dim), device=query.device, dtype=query.dtype)
    _decode_partials[(batch * kv_heads, splits)](
        query, key, value, position, partial, stats,
        GROUPS=groups, DIM=dim, CAPACITY=capacity, SPLITS=splits, CHUNK=chunk,
        SCALE=scale, BLOCK_M=max(16, triton.next_power_of_2(groups)), BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )
    _decode_merge[(batch * query_heads,)](
        partial, stats, out, GROUPS=groups, DIM=dim, SPLITS=splits,
        BLOCK_S=triton.next_power_of_2(splits), num_warps=4,
    )
    return out
