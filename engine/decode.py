"""Native Qwen layers with graph-stable decode storage.

Prefill exposes only the freshly written prompt to causal SDPA. Dense Triton
decode attention reads the valid prefix using a GPU position. Weights and KV
storage remain BF16; fused pointwise operations preserve native cast boundaries.
"""

import torch

from kernels.rmsnorm import add_rms_norm, rms_norm
from kernels.decode_attention import decode_attention
from kernels.linear import linear
from kernels.qk_rope import qk_rope_cache
from kernels.swiglu import swiglu
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
    mode, full persistent buffers in decode mode. Decode attention must restrict
    reads to the prefix ending at the device-side position.
    """

    def __init__(self, config, batch, capacity, device, dtype):
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
        shape = (batch, config.num_key_value_heads, capacity, head_dim)
        # Capture warmup runs before the first prefill. Keep its synthetic
        # prefix finite; real generation subsequently overwrites that prefix.
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
    """Full unpadded prefill, or one decode token; return last logits."""
    base = model.model
    hidden = base.embed_tokens(token_ids)
    residual = None
    for index, layer in enumerate(base.layers):
        last_token_only = cache.prefilling and token_ids.shape[1] > 1 and index == len(base.layers) - 1
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
            last_token_only=last_token_only,
        )[0]
        if last_token_only:
            # Final-layer historical MLP outputs never feed another layer or
            # the KV cache. All final-layer K/V entries were still computed.
            residual = residual[:, -1:, :]
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
    return linear(normalized, model.lm_head.weight)[:, 0, :]


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
        self.prompt_ids = torch.zeros((batch, prompt_length), dtype=torch.int64, device=self.device)
        # Native RoPE uses only the input's dtype/device, not its values.
        self.cos, self.sin = model.model.rotary_emb(
            weight.new_empty((1, 1, weight.shape[1])), self.positions.unsqueeze(0)
        )
        if batch <= 16:
            # Spend the bounded tuning budget on the largest weight traffic
            # first. These are synthetic, untimed shape probes, not KV state.
            layer = model.model.layers[0]
            for projection in (
                layer.mlp.gate_up_weight, layer.mlp.down_proj.weight,
                model.lm_head.weight, layer.self_attn.qkv_weight,
                layer.self_attn.o_proj.weight,
            ):
                linear(projection.new_zeros((batch, 1, projection.shape[1])), projection)
        # Likewise the launch widths of the small per-layer decode kernels.
        layer = model.model.layers[0]
        hidden = weight.new_zeros((batch, 1, weight.shape[1]))
        add_rms_norm(hidden, hidden, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
        swiglu(weight.new_zeros((batch, 1, layer.mlp.gate_up_weight.shape[0])))
        attention = layer.self_attn
        qk_rope_cache(
            weight.new_zeros((batch, 1, attention.qkv_weight.shape[0])), attention.q_norm, attention.k_norm,
            self.cos[:, :1, :].contiguous(), self.sin[:, :1, :].contiguous(), self.position,
            self.cache.keys[0], self.cache.values[0], model.config.num_attention_heads,
        )
        # Choose the dense attention interval layout here, on the ordinary
        # stream and at this shape's prompt length, never inside a capture.
        attention = model.model.layers[0].self_attn
        decode_attention(
            weight.new_zeros((batch, model.config.num_attention_heads, 1, attention.head_dim)),
            self.cache.keys[0], self.cache.values[0], self.position, attention.scaling,
        )
        # One host row and one event per output step. Decode replays are
        # enqueued ahead of the consumer; each row is an ordered snapshot of
        # ``token_ids`` taken between two replays on the same stream.
        try:
            self.host_tokens = torch.empty((output_length, batch), dtype=torch.int64, pin_memory=True)
        except RuntimeError:
            self.host_tokens = torch.empty((output_length, batch), dtype=torch.int64)
        self.events = [torch.cuda.Event() for _ in range(output_length)]
        self.enqueued = 0
        self.graph = None
        self.prefill_graph = None
        self.capture_prefill()
        if output_length > 1:
            self.capture()

    def decode(self):
        rope = (
            self.cos.index_select(1, self.position),
            self.sin.index_select(1, self.position),
        )
        logits = forward_last(
            self.model, self.token_ids, self.cache, self.position, rope
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

    def prefill_forward(self):
        length = self.shape[1]
        logits = forward_last(
            self.model,
            self.prompt_ids,
            self.cache,
            self.positions[:length],
            (self.cos[:, :length, :], self.sin[:, :length, :]),
        )
        self.token_ids.copy_(logits.argmax(dim=-1, keepdim=True))
        self.position.fill_(length)

    def capture_prefill(self):
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        self.cache.prefilling = True
        try:
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.prefill_forward()
            current.wait_stream(stream)
            torch.cuda.synchronize(self.device)
            # Separate memory pools: prefill/decode may be replayed in any
            # order during preparation, without aliasing graph temporaries.
            self.prefill_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.prefill_graph, stream=stream):
                self.prefill_forward()
            current.wait_stream(stream)
        finally:
            self.cache.prefilling = False

    def snapshot(self):
        step = self.enqueued
        self.host_tokens[step].copy_(self.token_ids[:, 0], non_blocking=True)
        self.events[step].record()
        self.enqueued = step + 1

    def advance(self, limit):
        """Enqueue decode steps until ``limit`` outputs are in flight or done."""
        while self.enqueued < limit:
            self.graph.replay()
            self.snapshot()

    def read(self, step):
        self.events[step].synchronize()
        return self.host_tokens[step].tolist()

    def prefill(self, prompt):
        # Steps left in flight by an abandoned generator precede this prefill
        # on the stream; it then overwrites every input they touched.
        self.enqueued = 0
        self.prompt_ids.copy_(prompt)
        self.prefill_graph.replay()
        self.snapshot()
        # Every prompt slot was overwritten. Old continuation slots remain
        # inaccessible until rewritten: decode excludes j > position on every
        # replay, including the first replay after warmup or another sample.
