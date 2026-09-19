"""Exact single-token grouped-query SDPA without repeating cached K/V heads."""

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def grouped_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    if query.shape[2] != 1 or attention_mask is None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )

    batch, query_heads, _, head_dim = query.shape
    kv_heads = key.shape[1]
    groups = query_heads // kv_heads
    # All query heads represent the SAME absolute token position. Treat each
    # group of heads as SDPA's query-row dimension, with one shared K/V head.
    # These rows are heads, not successive time steps: is_causal MUST be False.
    grouped_query = query.reshape(batch, kv_heads, groups, head_dim)
    output = torch.nn.functional.scaled_dot_product_attention(
        grouped_query, key, value,
        attn_mask=attention_mask, dropout_p=dropout,
        is_causal=False, scale=scaling,
    )
    return output.reshape(batch, 1, query_heads, head_dim), None
