"""Warmup-selected BF16 skinny projections, with native cuBLAS as a candidate.

All products accumulate in FP32 and round once to BF16. Split-K uses a separate
FP32 reduction, never atomics. Selection is cached by tensor shape before the
decode graph is captured; prefill's large matrix products remain native.
"""

import statistics
import time

import torch
from torch.nn import functional as F
import triton
import triton.language as tl


@triton.jit
def _gemv(
    x_ptr, weight_ptr, out_ptr,
    N: tl.constexpr, K: tl.constexpr, SPLITS: tl.constexpr,
    CHUNK: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    columns = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for start in range(split * CHUNK, (split + 1) * CHUNK, BLOCK_K):
        k = start + columns
        x = tl.load(x_ptr + k, k < K, other=0).to(tl.float32)
        weight = tl.load(
            weight_ptr + rows[:, None] * K + k[None, :],
            (rows[:, None] < N) & (k[None, :] < K), other=0,
        ).to(tl.float32)
        acc += weight * x[None, :]
    result = tl.sum(acc, axis=1)
    tl.store(out_ptr + split * N + rows, result, rows < N)


@triton.jit
def _skinny_gemm(
    x_ptr, weight_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, 16)
    columns = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    reduction = tl.arange(0, BLOCK_K)
    acc = tl.zeros((16, BLOCK_N), tl.float32)
    for start in range(split * CHUNK, (split + 1) * CHUNK, BLOCK_K):
        k = start + reduction
        x = tl.load(
            x_ptr + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K), other=0,
        )
        weight = tl.load(
            weight_ptr + columns[None, :] * K + k[:, None],
            (columns[None, :] < N) & (k[:, None] < K), other=0,
        )
        acc = tl.dot(x, weight, acc)
    tl.store(
        out_ptr + split * M * N + rows[:, None] * N + columns[None, :], acc,
        (rows[:, None] < M) & (columns[None, :] < N),
    )


@triton.jit
def _merge_projection(
    partial_ptr, out_ptr, COUNT: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, BLOCK_S)
    values = tl.load(
        partial_ptr + splits[:, None] * COUNT + offsets[None, :],
        (splits[:, None] < SPLITS) & (offsets[None, :] < COUNT), other=0,
    )
    tl.store(out_ptr + offsets, tl.sum(values, axis=0), offsets < COUNT)


def _project(x, weight, config):
    kind, block_n, block_k, splits = config
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    partial = out if splits == 1 else torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    chunk = triton.cdiv(k, splits * block_k) * block_k
    if kind == "gemv":
        _gemv[(triton.cdiv(n, block_n), splits)](
            x, weight, partial, N=n, K=k, SPLITS=splits, CHUNK=chunk,
            BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4,
        )
    else:
        _skinny_gemm[(triton.cdiv(n, block_n), splits)](
            x, weight, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
            BLOCK_N=block_n, BLOCK_K=block_k, num_warps=4, num_stages=2,
        )
    if splits > 1:
        _merge_projection[(triton.cdiv(m * n, 512),)](
            partial, out, COUNT=m * n, SPLITS=splits,
            BLOCK_S=triton.next_power_of_2(splits), BLOCK=512, num_warps=4,
        )
    return out


def _cold_graph_time(fn, flush):
    # CPU dispatch is absent from decode. Benchmark a graph, with >H100 L2
    # bytes touched before each projection, to approximate its cold weights.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            flush.zero_()
            fn()
    graph.replay()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 8)
    return statistics.median(times)


_CHOICES = {}
_TUNING_DEADLINE = None


def _choose(x, weight):
    global _TUNING_DEADLINE
    if _TUNING_DEADLINE is None:
        _TUNING_DEADLINE = time.monotonic() + 12.0
    if time.monotonic() >= _TUNING_DEADLINE:
        return None
    m, k = x.shape
    n = weight.shape[0]
    configs = []
    # Keep the initial search small: every workload starts a fresh process and
    # compilation must fit the event's load/warmup and whole-run budgets.
    for block_n, block_k in ((64, 128),):
        splits = min(8, triton.next_power_of_2(triton.cdiv(512, triton.cdiv(n, block_n))))
        configs.append(("gemm", block_n, block_k, splits))
    if m == 1:
        configs.append(("gemv", 8, 512, 1))
    # Private generator: tuning must not change any caller's RNG state.
    generator = torch.Generator(device=x.device).manual_seed(1729)
    probe = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    reference = F.linear(probe, weight)
    flush = torch.empty(32 * 1024 * 1024, device=x.device, dtype=torch.int32)
    native_ms = _cold_graph_time(lambda: F.linear(x, weight), flush)
    best_ms, best = native_ms, None
    for config in configs:
        if time.monotonic() >= _TUNING_DEADLINE:
            break
        actual = _project(probe, weight, config)
        # Reject a kernel that fails an operator sanity check. Full-model
        # correctness still comes from the platform's own-prefix replay.
        close = (actual.float() - reference.float()).abs() <= reference.float().abs() * 0.016 + 0.001
        if not bool(close.all()):
            continue
        elapsed = _cold_graph_time(lambda: _project(x, weight, config), flush)
        if elapsed < best_ms * 0.985:
            best_ms, best = elapsed, config
    if best is not None:
        # Recheck after compilation/tuning so GPU clock ramp-up cannot make a
        # later candidate look faster than an initially cold native baseline.
        native_ms = _cold_graph_time(lambda: F.linear(x, weight), flush)
        best_ms = _cold_graph_time(lambda: _project(x, weight, best), flush)
        if best_ms >= native_ms * 0.985:
            best_ms, best = native_ms, None
    print(f"BF16 projection warmup: backend={best or 'cublas'} cold_graph_ratio={best_ms / native_ms:.3f}", flush=True)
    return best


def linear(x, weight):
    rows = x.numel() // x.shape[-1]
    if rows > 16 or x.dtype != torch.bfloat16 or not weight.is_contiguous():
        return F.linear(x, weight)
    flat = x.reshape(rows, x.shape[-1]).contiguous()
    key = (x.device, rows, weight.shape[0], weight.shape[1])
    if key not in _CHOICES:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("projection selection must finish during eager warmup")
        _CHOICES[key] = _choose(flat, weight)
    choice = _CHOICES[key]
    if choice is None:
        return F.linear(x, weight)
    return _project(flat, weight, choice).reshape(*x.shape[:-1], weight.shape[0])
