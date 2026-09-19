"""Warmup-only choice among launch options for one small fixed-shape kernel.

The options run the same kernel source; only the launch width differs. Each
choice is made once per key on the ordinary stream and is frozen before any
CUDA graph capture, so measured samples never compile or re-select.
"""

import statistics
import time

import torch

_BEST = {}
_DEADLINE = None
_PROCESS_SECONDS = 12.0


def _graph_time(run):
    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32):
            run()
    graph.replay()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(9):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def pick(key, default, options, launch):
    """Return the fastest of ``options`` for ``launch(option)``, else ``default``."""
    global _DEADLINE
    if key in _BEST:
        return _BEST[key]
    if torch.cuda.is_current_stream_capturing():
        return default
    if _DEADLINE is None:
        _DEADLINE = time.monotonic() + _PROCESS_SECONDS
    best, best_ms = default, None
    if time.monotonic() < _DEADLINE:
        best_ms = _graph_time(lambda: launch(default))
        for option in options:
            if option == default or time.monotonic() >= _DEADLINE:
                continue
            elapsed = _graph_time(lambda: launch(option))
            if elapsed < best_ms * 0.97:
                best, best_ms = option, elapsed
    _BEST[key] = best
    return best
