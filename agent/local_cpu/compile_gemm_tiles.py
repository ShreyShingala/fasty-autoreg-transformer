"""kernels/gemm.py through the real kernels.linear._project: compile for cuda:90 and emulate.

A recorder stands in for the two kernels, so the grid and constants checked here
are exactly what `_project` would launch. For each launch: (1) offline-compile
with those constants; (2) replay the kernel's pointer arithmetic in numpy (masks
present only where the EVEN_* flags keep them) and require in-bounds loads,
every output cell of every split written exactly once, and the FP64 product.
"""
import itertools

import numpy as np
import torch

from offline_compile import compile_kernel
from kernels import linear


class Recorder:
    def __init__(self, kernel):
        self.kernel, self.calls = kernel, []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, kwargs))


def emulate(kind, grid, c, x, w):
    m, n, k = c["M"], c["N"], c["K"]
    out = np.full((c["SPLITS"], m, n), np.nan)
    writes = np.zeros((c["SPLITS"], m, n), int)
    for tile, split in itertools.product(range(grid[0]), range(grid[1])):
        rows = np.arange(c["BLOCK_M"])
        cols = tile * c["BLOCK_N"] + np.arange(c["BLOCK_N"])
        row_ok = rows < m if not c["EVEN_M"] else np.ones_like(rows, bool)
        col_ok = cols < n if not c["EVEN_N"] else np.ones_like(cols, bool)
        assert rows[row_ok].max() < m and cols[col_ok].max() < n, "unmasked out-of-bounds lane"
        acc = np.zeros((c["BLOCK_M"], c["BLOCK_N"]))
        for step in range(c["CHUNK"] // c["BLOCK_K"]):
            ks = split * c["CHUNK"] + step * c["BLOCK_K"] + np.arange(c["BLOCK_K"])
            k_ok = ks < k if not c["EVEN_K"] else np.ones_like(ks, bool)
            assert not k_ok.any() or ks[k_ok].max() < k, "unmasked out-of-bounds k"
            xt = np.where(row_ok[:, None] & k_ok[None, :], x[np.minimum(rows, m - 1)][:, np.minimum(ks, k - 1)], 0.0)
            wt = np.where(col_ok[:, None] & k_ok[None, :], w[np.minimum(cols, n - 1)][:, np.minimum(ks, k - 1)], 0.0)
            acc += xt @ wt.T
        for i, j in itertools.product(rows[row_ok], range(c["BLOCK_N"])):
            if col_ok[j]:
                out[split, i, cols[j]] = acc[i, j]
                writes[split, i, cols[j]] += 1
    assert (writes == 1).all(), "a cell was not written exactly once"
    return out.sum(0)


def launches(m, n, k, config):
    recorders = {name: Recorder(getattr(linear, name)) for name in ("_exact_gemm", "_trans_gemm")}
    for name, recorder in recorders.items():
        setattr(linear, name, recorder)
    try:
        x = torch.zeros((m, k), dtype=torch.bfloat16)
        w = torch.zeros((n, k), dtype=torch.bfloat16)
        linear._project(x, w, config, split_ok=True)
    finally:
        for name, recorder in recorders.items():
            setattr(linear, name, recorder.kernel)
    (name, recorder), = [(a, r) for a, r in recorders.items() if r.calls]
    (grid, kwargs), = recorder.calls
    return recorder.kernel, grid, kwargs


failures = 0
# Real shapes: qkv, o, gate_up, down at verify-block row counts (even and ragged M).
for m, (n, k) in itertools.product((5, 32), ((6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728))):
    for config in linear._candidates(m, n, k):
        if config[0] not in ("exact", "trans"):
            continue
        kernel, grid, kwargs = launches(m, n, k, config)
        constants = {key: value for key, value in kwargs.items() if key not in ("num_warps", "num_stages")}
        assert constants["SPLITS"] * constants["CHUNK"] >= k and (constants["SPLITS"] - 1) * constants["CHUNK"] < k
        try:
            ptr = "*bf16" if constants["SPLITS"] == 1 else "*fp32"
            compile_kernel(kernel, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": ptr}, constants,
                           num_warps=kwargs["num_warps"], num_stages=kwargs["num_stages"])
            print("compiled", m, n, k, config, grid, {key: constants[key] for key in ("SPLITS", "CHUNK", "EVEN_M", "EVEN_N", "EVEN_K", "WIDE")})
        except Exception as error:  # noqa: BLE001
            failures += 1
            print("COMPILE FAILED", m, n, k, config, repr(error)[:300])

# Small shapes through the same _project constants: even and ragged on every axis.
rng = np.random.default_rng(0)
for m, n, k in ((16, 128, 256), (5, 128, 256), (16, 100, 256), (32, 192, 384), (7, 70, 300), (16, 64, 128)):
    for kind in ("exact", "trans"):
        for splits in (1, 2, 3):
            config = (kind, 64, 128, splits, 4)
            if (splits - 1) * (-(-k // (splits * 128)) * 128) >= k:
                continue  # a dead split: exact_splits never produces one
            kernel, grid, kwargs = launches(m, n, k, config)
            x, w = rng.standard_normal((m, k)), rng.standard_normal((n, k))
            got = emulate(kind, grid, kwargs, x, w)
            if not np.allclose(got, x @ w.T):
                failures += 1
                print("EMULATION MISMATCH", m, n, k, config)
print("failures", failures)
