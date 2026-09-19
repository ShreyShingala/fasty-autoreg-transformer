"""Decode-only Q/K head RMSNorm, RoPE, and in-place KV-cache update."""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_rope_cache(
    packed, q_weight, k_weight, cos, sin, position, query, keys, values,
    Q_HEADS: tl.constexpr, KV_HEADS: tl.constexpr, DIM: tl.constexpr,
    CAPACITY: tl.constexpr, Q_EPS: tl.constexpr, K_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < DIM
    paired = (col + DIM // 2) % DIM
    packed_row = batch * (Q_HEADS + 2 * KV_HEADS) * DIM
    offset = packed_row + head * DIM
    x = tl.load(packed + offset + col, valid, other=0).to(tl.float32)
    x_pair = tl.load(packed + offset + paired, valid, other=0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / DIM
    eps = tl.where(head < Q_HEADS, Q_EPS, K_EPS)
    inv_std = tl.rsqrt(variance + eps)
    if head < Q_HEADS:
        gain = tl.load(q_weight + col, valid, other=0).to(tl.float32)
        gain_pair = tl.load(q_weight + paired, valid, other=0).to(tl.float32)
    else:
        gain = tl.load(k_weight + col, valid, other=0).to(tl.float32)
        gain_pair = tl.load(k_weight + paired, valid, other=0).to(tl.float32)

    # Native norm: FP32 normalization -> BF16 -> learned gain -> BF16.
    normalized = (x * inv_std).to(tl.bfloat16).to(tl.float32)
    normalized_pair = (x_pair * inv_std).to(tl.bfloat16).to(tl.float32)
    weighted = (normalized * gain).to(tl.bfloat16).to(tl.float32)
    weighted_pair = (normalized_pair * gain_pair).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(col < DIM // 2, -weighted_pair, weighted_pair)
    cosine = tl.load(cos + col, valid, other=0).to(tl.float32)
    sine = tl.load(sin + col, valid, other=0).to(tl.float32)
    # Native RoPE rounds BOTH products before their BF16 addition. Keeping
    # these products in FP32 and rounding only the sum changes the function.
    direct = (weighted * cosine).to(tl.bfloat16).to(tl.float32)
    turn = (rotated * sine).to(tl.bfloat16).to(tl.float32)
    result = direct + turn
    if head < Q_HEADS:
        tl.store(query + (batch * Q_HEADS + head) * DIM + col, result, valid)
    else:
        kv_head = head - Q_HEADS
        pos = tl.load(position).to(tl.int64)
        cache_offset = ((batch * KV_HEADS + kv_head) * CAPACITY + pos) * DIM + col
        tl.store(keys + cache_offset, result, valid)
        value_offset = packed_row + (Q_HEADS + KV_HEADS + kv_head) * DIM + col
        value = tl.load(packed + value_offset, valid, other=0)
        tl.store(values + cache_offset, value, valid)


def qk_rope_cache(packed, q_norm, k_norm, cos, sin, position, keys, values, q_heads):
    """One BF16 token per batch; contiguous packed QKV and [B,Hkv,C,D] caches.

    ``cos``/``sin`` are native BF16 [1,1,D] values shared by every batch row.
    ``position`` is a CUDA int64 [1] tensor. Return contiguous Q [B,Hq,1,D];
    mutate only K/V at that position, leaving every other cache slot unchanged.
    """
    batch, kv_heads, capacity, dim = keys.shape
    assert packed.dtype == keys.dtype == values.dtype == torch.bfloat16
    assert packed.is_contiguous() and keys.is_contiguous() and values.is_contiguous()
    assert packed.shape == (batch, 1, (q_heads + 2 * kv_heads) * dim)
    assert values.shape == keys.shape and dim % 2 == 0
    assert cos.numel() == sin.numel() == dim
    assert cos.is_contiguous() and sin.is_contiguous()
    assert position.shape == (1,) and position.dtype == torch.int64
    query = torch.empty((batch, q_heads, 1, dim), device=packed.device, dtype=packed.dtype)
    _qk_rope_cache[(batch, q_heads + kv_heads)](
        packed, q_norm.weight, k_norm.weight, cos, sin, position, query, keys, values,
        Q_HEADS=q_heads, KV_HEADS=kv_heads, DIM=dim, CAPACITY=capacity,
        Q_EPS=q_norm.variance_epsilon, K_EPS=k_norm.variance_epsilon,
        BLOCK=triton.next_power_of_2(dim), num_warps=4,
    )
    return query
