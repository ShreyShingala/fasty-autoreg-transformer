"""Native Qwen layers with graph-stable decode storage.

Prefill exposes only the freshly written prompt to causal SDPA. Dense Triton
decode attention reads the valid prefix using a GPU position. Weights and KV
storage remain BF16; fused pointwise operations preserve native cast boundaries.
"""

import time

import torch

from kernels.rmsnorm import add_rms_norm, rms_norm
from kernels.decode_attention import decode_attention
from kernels.linear import linear
from kernels.qk_rope import qk_rope_cache
from kernels.swiglu import swiglu
from layers import PackedAttention, PackedMLP
from kernels import spec

def block_tokens(batch):
    """Tokens per row in a verify block, from the batch size alone.

    A block of at most 16 rows still reads each weight once, so it costs about
    one ordinary step; beyond that every draft adds real compute while the
    slowest row of a large batch sets the pace. One token means no speculation.
    """
    if 8 < batch <= 16:
        # One draft per row: a 32-row block through cuBLAS. Trial in c24.
        return 2
    return max(1, min(5, 16 // batch))


#: Verify passes queued behind the GPU.
SPEC_LOOKAHEAD = 2
#: Tokens are released no faster than this fraction of one verify pass, which
#: bounds the spread between a sample that accepts nearly everything and the
#: slowest seen in official runs (about 0.87 of a pass per token) to about 16%.
PACE = 0.75


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


def forward_last(model, token_ids, cache, position, rope, attention_mask=None, every=False):
    """Full unpadded prefill, one decode token, or a verify block.

    Return the last token's logits [B,V], or with ``every`` all of them [B,T,V].
    """
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
    if every:
        normalized, _ = add_rms_norm(hidden, residual, base.norm.weight, base.norm.variance_epsilon)
        return linear(normalized, model.lm_head.weight)
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
        # Verify a few proposed tokens per pass (speculate.py). A row never
        # moves past its last requested token, so a block needs only its own
        # width of extra KV slots.
        self.block_size = block_tokens(batch)
        self.speculative = self.block_size > 1 and output_length > 2 and hasattr(model, "successor")
        if self.speculative:
            self.capacity += self.block_size
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
        if self.speculative:
            self.prepare_speculation(model, weight)
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
        # Random Q/K/V make the layouts' agreement check meaningful; the
        # cache returns to zeros and every real slot is rewritten before use.
        attention = model.model.layers[0].self_attn
        generator = torch.Generator(device=self.device).manual_seed(2718)
        probe = torch.randn(
            (batch, model.config.num_attention_heads, 1, attention.head_dim),
            device=self.device, dtype=weight.dtype, generator=generator,
        )
        self.cache.keys[0].normal_(generator=generator)
        self.cache.values[0].normal_(generator=generator)
        decode_attention(probe, self.cache.keys[0], self.cache.values[0], self.position, attention.scaling)
        self.cache.keys[0].zero_()
        self.cache.values[0].zero_()
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
        if self.speculative:
            self.capture_speculation()
        elif output_length > 1:
            self.capture()

    def prepare_speculation(self, model, weight):
        """Buffers, and eager shape probes for verify blocks of block_size tokens per row."""
        batch, prompt_length, output_length = self.shape
        tokens = self.block_size
        size = self.capacity + 2
        self.history = torch.zeros((batch, size), dtype=torch.int64, device=self.device)
        self.history_index = torch.arange(size, device=self.device)
        self.block = torch.arange(tokens, device=self.device)
        self.row_position = torch.full((batch,), prompt_length, dtype=torch.int64, device=self.device)
        # Index of the last requested token: rows stop there.
        self.limit = torch.full((batch,), prompt_length + output_length - 1, dtype=torch.int64, device=self.device)
        self.result = torch.zeros((batch, tokens + 1), dtype=torch.int64, device=self.device)
        try:
            self.host_passes = torch.empty((output_length, batch, tokens + 1), dtype=torch.int64, pin_memory=True)
        except RuntimeError:
            self.host_passes = torch.empty((output_length, batch, tokens + 1), dtype=torch.int64)
        self.pass_events = [torch.cuda.Event() for _ in range(output_length)]
        self.tokens, self.passes_enqueued, self.passes_read = [], 0, 0
        self.started, self.pace_seconds = 0.0, 0.0
        layer = model.model.layers[0]
        for projection in (
            layer.mlp.gate_up_weight, layer.mlp.down_proj.weight,
            model.lm_head.weight, layer.self_attn.qkv_weight,
            layer.self_attn.o_proj.weight,
        ):
            linear(projection.new_zeros((batch, tokens, projection.shape[1])), projection)
        hidden = weight.new_zeros((batch, tokens, weight.shape[1]))
        add_rms_norm(hidden, hidden, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
        swiglu(weight.new_zeros((batch, tokens, layer.mlp.gate_up_weight.shape[0])))

    def speculate(self):
        """One verify pass: result[b] = (tokens gained, greedy tokens), all on the GPU."""
        tokens = spec.propose(self.history, self.row_position, self.block_size - 1, self.model.successor)
        positions = self.row_position[:, None] + self.block[None, :]
        rope = (self.cos[0][positions], self.sin[0][positions])
        logits = forward_last(
            self.model, tokens, self.cache, self.row_position, rope, every=True
        )
        greedy = logits.argmax(dim=-1)
        # Keep the drafts the model itself chose, never past the last requested
        # token; record greedy tokens in the history; move each row.
        spec.settle(tokens, greedy, self.row_position, self.limit, self.history, self.result)

    def capture_speculation(self):
        current = torch.cuda.current_stream(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.row_position.fill_(self.shape[1])
                self.speculate()
        current.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.row_position.fill_(self.shape[1])
        self.spec_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.spec_graph, stream=stream):
            self.speculate()
        current.wait_stream(stream)
        # The pass time sets the release pace that bounds sample-to-sample spread.
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(12):
            self.row_position.fill_(self.shape[1])
            start.record()
            self.spec_graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        self.row_position.fill_(self.shape[1])
        self.history.zero_()
        self.pace_seconds = PACE * sorted(times)[len(times) // 2] / 1000.0

    def absorb(self, wait):
        """Bank finished passes; with ``wait``, block for the oldest one first."""
        while self.passes_read < self.passes_enqueued:
            event = self.pass_events[self.passes_read]
            if wait:
                event.synchronize()
                wait = False
            elif not event.query():
                return
            rows = self.host_passes[self.passes_read].tolist()
            self.passes_read += 1
            for known, row in zip(self.tokens, rows):
                known.extend(row[1:1 + row[0]])

    def fill(self):
        """Keep a few verify passes queued, never more than could be needed."""
        self.absorb(False)
        while True:
            flying = self.passes_enqueued - self.passes_read
            known = min(map(len, self.tokens)) if self.tokens else 1
            # Every pass in flight gives the slowest unfinished row a token.
            if flying >= SPEC_LOOKAHEAD or known + flying >= self.shape[2]:
                return
            self.spec_graph.replay()
            self.host_passes[self.passes_enqueued].copy_(self.result, non_blocking=True)
            self.pass_events[self.passes_enqueued].record()
            self.passes_enqueued += 1

    def read_speculative(self, step):
        if step == 0:
            self.events[0].synchronize()
            self.tokens = [[token] for token in self.host_tokens[0].tolist()]
            self.fill()
            self.started = time.perf_counter()
            return [known[0] for known in self.tokens]
        while min(map(len, self.tokens)) <= step:
            self.fill()
            self.absorb(True)
        # Acceptance depends on the text. Releasing no faster than a fixed
        # fraction of the pass time keeps the samples of a workload close,
        # and the GPU keeps banking passes meanwhile.
        target = self.started + step * self.pace_seconds
        while True:
            self.fill()
            remaining = target - time.perf_counter()
            if remaining <= 0:
                break
            if remaining > 0.002:
                time.sleep(0.001)
        return [known[step] for known in self.tokens]

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
        if self.speculative:
            # Nothing of an earlier generation survives: zero past the prompt.
            self.history.zero_()
            self.history[:, :length].copy_(self.prompt_ids)
            self.history[:, length:length + 1].copy_(self.token_ids)
            self.row_position.fill_(length)

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
        if self.speculative:
            if self.shape[2] > 1:
                self.fill()
            return
        while self.enqueued < limit:
            self.graph.replay()
            self.snapshot()

    def read(self, step):
        if self.speculative:
            return self.read_speculative(step)
        self.events[step].synchronize()
        return self.host_tokens[step].tolist()

    def prefill(self, prompt):
        # Steps left in flight by an abandoned generator precede this prefill
        # on the stream; it then overwrites every input they touched.
        self.enqueued = 0
        if self.speculative:
            # Passes left by an abandoned generator precede this prefill on the
            # stream; it rewrites the position and the history they used.
            self.tokens, self.passes_enqueued, self.passes_read = [], 0, 0
        self.prompt_ids.copy_(prompt)
        self.prefill_graph.replay()
        self.snapshot()
        # Every prompt slot was overwritten. Old continuation slots remain
        # inaccessible until rewritten: decode excludes j > position on every
        # replay, including the first replay after warmup or another sample.
