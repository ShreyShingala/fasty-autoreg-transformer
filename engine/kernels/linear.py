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

from kernels import packed as packing


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


def _merge(partial, out, count, splits):
    _merge_projection[(triton.cdiv(count, 512),)](
        partial, out, COUNT=count, SPLITS=splits,
        BLOCK_S=triton.next_power_of_2(splits), BLOCK=512, num_warps=4,
    )


def _project(x, weight, config):
    kind, block_n, block_k, splits, warps = config
    if kind in ("pgemv", "pgemm"):
        return packing.project(x, packing.lookup(weight), config, _merge)
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    partial = out if splits == 1 else torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    chunk = triton.cdiv(k, splits * block_k) * block_k
    if kind == "gemv":
        _gemv[(triton.cdiv(n, block_n), splits)](
            x, weight, partial, N=n, K=k, SPLITS=splits, CHUNK=chunk,
            BLOCK_N=block_n, BLOCK_K=block_k, num_warps=warps,
        )
    else:
        _skinny_gemm[(triton.cdiv(n, block_n), splits)](
            x, weight, partial, M=m, N=n, K=k, SPLITS=splits, CHUNK=chunk,
            BLOCK_N=block_n, BLOCK_K=block_k, num_warps=warps, num_stages=2,
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
_PROCESS_SECONDS = 80.0
_SHAPE_SECONDS = 16.0


def _candidates(m, n, k, packed):
    """Layouts in order of prior plausibility; the deadline truncates the tail."""
    def gemm(block_n, block_k):
        splits = min(8, triton.next_power_of_2(triton.cdiv(512, triton.cdiv(n, block_n))))
        return ("gemm", block_n, block_k, splits, 4)

    configs = []
    if packed:
        # Lossless 12-bit planes first: they carry 25% less memory traffic.
        if m == 1:
            configs += [("pgemv", 8, 512, 1, 4), ("pgemv", 16, 256, 1, 4)]
        else:
            configs += [("pgemm",) + gemm(64, 128)[1:], ("pgemm",) + gemm(128, 128)[1:]]
    configs.append(gemm(64, 128))
    if m == 1:
        configs += [("gemv", 8, 512, 1, 4), ("gemv", 16, 256, 1, 4), ("gemv", 4, 1024, 1, 4)]
        if k >= 8192:
            # A long reduction is serial inside one program; split it.
            configs.append(("gemv", 8, 512, 4, 4))
        configs.append(("gemv", 8, 1024, 1, 8))
    else:
        configs += [gemm(32, 256), gemm(128, 128), gemm(32, 128)]
    return configs


def _choose(x, weight):
    global _TUNING_DEADLINE
    now = time.monotonic()
    if _TUNING_DEADLINE is None:
        _TUNING_DEADLINE = now + _PROCESS_SECONDS
    if now >= _TUNING_DEADLINE:
        return None, None
    # Every workload is a fresh process: bound each shape and the process so
    # compilation fits the load/warmup and whole-run budgets, and so one slow
    # shape cannot leave the later projections unmeasured.
    shape_deadline = min(_TUNING_DEADLINE, now + _SHAPE_SECONDS)
    m, k = x.shape
    n = weight.shape[0]
    packed = packing.lookup(weight)
    configs = _candidates(m, n, k, packed is not None)
    # Private generator: tuning must not change any caller's RNG state.
    generator = torch.Generator(device=x.device).manual_seed(1729)
    probe = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    reference = F.linear(probe, weight)
    flush = torch.empty(32 * 1024 * 1024, device=x.device, dtype=torch.int32)
    native_ms = _cold_graph_time(lambda: F.linear(x, weight), flush)
    # Track the best layout overall and the best one that reads plain BF16
    # weights: a matrix that could not be packed falls back to the latter.
    best_ms, best = native_ms, None
    plain_ms, plain = native_ms, None
    for config in configs:
        if time.monotonic() >= shape_deadline:
            break
        is_packed = config[0] in ("pgemv", "pgemm")
        try:
            actual = _project(probe, weight, config)
        except Exception as error:  # a layout that cannot compile is not a candidate
            print(f"projection layout {config} skipped: {error!r}", flush=True)
            continue
        # Reject a kernel that fails an operator sanity check. Full-model
        # correctness still comes from the platform's own-prefix replay.
        close = (actual.float() - reference.float()).abs() <= reference.float().abs() * 0.016 + 0.001
        if not bool(close.all()):
            continue
        if is_packed and not packing.exact_columns(weight, packed, config, _merge, m):
            continue
        elapsed = _cold_graph_time(lambda: _project(x, weight, config), flush)
        if elapsed < best_ms * 0.985:
            best_ms, best = elapsed, config
        if not is_packed and elapsed < plain_ms * 0.985:
            plain_ms, plain = elapsed, config
    if best is not None:
        # Recheck after compilation/tuning so GPU clock ramp-up cannot make a
        # later candidate look faster than an initially cold native baseline.
        native_ms = _cold_graph_time(lambda: F.linear(x, weight), flush)
        best_ms = _cold_graph_time(lambda: _project(x, weight, best), flush)
        if best_ms >= native_ms * 0.985:
            best_ms, best = native_ms, None
    if plain is not None and plain is not best:
        if _cold_graph_time(lambda: _project(x, weight, plain), flush) >= native_ms * 0.985:
            plain = None
    print(f"BF16 projection warmup: backend={best or 'cublas'} cold_graph_ratio={best_ms / native_ms:.3f}", flush=True)
    return best, plain


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
    choice, plain = _CHOICES[key]
    if choice is not None and choice[0] in ("pgemv", "pgemm") and packing.lookup(weight) is None:
        choice = plain
    if choice is None:
        return F.linear(x, weight)
    return _project(flat, weight, choice).reshape(*x.shape[:-1], weight.shape[0])
