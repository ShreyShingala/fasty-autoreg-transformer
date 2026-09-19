"""Native Qwen layers with graph-stable decode storage.

Prefill exposes only the freshly written prompt to causal SDPA. Decode exposes
the full cache capacity and supplies an explicit mask based on a GPU position.
All weights, matrix products, RoPE arithmetic and attention remain BF16/native.
"""

import torch

from kernels.rmsnorm import add_rms_norm, rms_norm
from layers import PackedAttention, PackedMLP


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)


def optimize_model(model):
    base = model.model
    base.norm = FusedRMSNorm(base.norm)
    for layer in base.layers:
        layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
        layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
        layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)
        layer.self_attn = PackedAttention(layer.self_attn)
        layer.mlp = PackedMLP(layer.mlp)


class KVCache:
    """Layer-local BF16 [batch, kv_heads, capacity, head_dim] buffers.

    Only decoder layers see this object; the generic Transformers model/cache
    dispatcher is bypassed. ``update`` returns prompt-sized inputs in prefill
    mode, full persistent buffers in decode mode. The latter requires a mask.
    """

    def __init__(self, config, batch, capacity, device, dtype):
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        shape = (batch, config.num_key_value_heads, capacity, head_dim)
        # Masked SDPA can still load unused slots. Initialize them to finite
        # values: multiplying masked weights by uninitialized NaNs is unsafe.
        self.keys = [
            torch.zeros(shape, dtype=dtype, device=device)
            for _ in range(config.num_hidden_layers)
        ]
        self.values = [torch.zeros_like(key) for key in self.keys]
        self.prefilling = False

    def update(self, key, value, layer_idx, cache_kwargs):
        k_cache = self.keys[layer_idx]
        v_cache = self.values[layer_idx]
        if self.prefilling:
            length = key.shape[2]
            k_cache[:, :, :length, :].copy_(key)
            v_cache[:, :, :length, :].copy_(value)
            return key, value
        position = cache_kwargs["cache_position"]
        k_cache.index_copy_(2, position, key)
        v_cache.index_copy_(2, position, value)
        return k_cache, v_cache


def forward_last(model, token_ids, cache, position, rope, attention_mask=None):
    """Full unpadded prefill, or one masked decode token; return last logits."""
    base = model.model
    hidden = base.embed_tokens(token_ids)
    residual = None
    for layer in base.layers:
        if residual is None:
            residual = hidden
            normalized = layer.input_layernorm(hidden)
        else:
            # Complete the previous layer's MLP residual in the next layer's
            # input norm. The stored sum rounds to BF16 before normalization.
            normalized, residual = add_rms_norm(
                hidden, residual, layer.input_layernorm.weight,
                layer.input_layernorm.variance_epsilon,
            )
        attention = layer.self_attn(
            normalized,
            attention_mask=attention_mask,
            position_ids=position.unsqueeze(0),
            past_key_value=cache,
            use_cache=True,
            cache_position=position,
            position_embeddings=rope,
        )[0]
        normalized, residual = add_rms_norm(
            attention, residual, layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
        )
        hidden = layer.mlp(normalized)
    # RMSNorm acts independently on each token; earlier final states are unused.
    normalized, _ = add_rms_norm(
        hidden[:, -1:, :], residual[:, -1:, :],
        base.norm.weight, base.norm.variance_epsilon,
    )
    return model.lm_head(normalized)[:, 0, :]


class DecodeState:
    def __init__(self, model, shape):
        self.model = model
        self.shape = shape
        batch, prompt_length, output_length = shape
        weight = model.model.embed_tokens.weight
        self.device = weight.device
        self.capacity = prompt_length + output_length
        self.cache = KVCache(
            model.config, batch, self.capacity, self.device, weight.dtype
        )
        self.positions = torch.arange(self.capacity, device=self.device)
        self.position = torch.full(
            (1,), prompt_length, dtype=torch.int64, device=self.device
        )
        self.token_ids = torch.zeros((batch, 1), dtype=torch.int64, device=self.device)
        # Native RoPE uses only the input's dtype/device, not its values.
        self.cos, self.sin = model.model.rotary_emb(
            weight.new_empty((1, 1, weight.shape[1])), self.positions.unsqueeze(0)
        )
        self.graph = None
        if output_length > 1:
            self.capture()

    def decode(self):
        rope = (
            self.cos.index_select(1, self.position),
            self.sin.index_select(1, self.position),
        )
        mask = (self.positions <= self.position).view(1, 1, 1, self.capacity)
        logits = forward_last(
            self.model, self.token_ids, self.cache, self.position, rope, mask
        )
        self.token_ids.copy_(logits.argmax(dim=-1, keepdim=True))
        self.position.add_(1)

    def capture(self):
        # Compile Triton and initialize CUDA libraries before capture. Side
        # stream ordering follows PyTorch's CUDA graph warmup requirements.
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.position.fill_(self.shape[1])
                self.token_ids.zero_()
                self.decode()
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.position.fill_(self.shape[1])
        self.token_ids.zero_()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.decode()
        current.wait_stream(stream)

    def prefill(self, prompt):
        length = self.shape[1]
        self.cache.prefilling = True
        try:
            logits = forward_last(
                self.model,
                prompt,
                self.cache,
                self.positions[:length],
                (self.cos[:, :length, :], self.sin[:, :length, :]),
            )
        finally:
            self.cache.prefilling = False
        self.token_ids.copy_(logits.argmax(dim=-1, keepdim=True))
        self.position.fill_(length)
        # Every prompt slot was overwritten. Old continuation slots remain
        # inaccessible until rewritten: decode masks j > position on every
        # replay, including the first replay after warmup or another sample.
