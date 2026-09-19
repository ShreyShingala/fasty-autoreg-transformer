"""The "tma" projection kind through the real kernels.linear._project, compiled for cuda:90.

A recorder stands in for ``_tma_gemm`` and a stub for the descriptor, so the grid
and constants are what ``_project`` would launch on the GPU. For each real shape
the TTGIR must hold the TMA copy op and the PTX the bulk-tensor instruction: that
is the evidence the descriptor load was lowered to the Hopper TMA unit rather
than to ordinary global loads. Also checks that without a descriptor the same
config launches ``_trans_gemm`` (the fail-safe), and that strict mode raises.
"""
import itertools

import torch

from offline_compile import compile_kernel
from kernels import linear


class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


def record(m, n, k, config, descriptor, strict=False):
    tma, trans, real = Recorder(), Recorder(), (linear._tma_gemm, linear._trans_gemm, linear._tma_descriptor)
    linear._tma_gemm, linear._trans_gemm = tma, trans
    # The real one needs CUDA (on this CPU it retires the kind, which is its own fail-safe).
    stub = torch.zeros(linear.TMA_SIZE, dtype=torch.int8) if descriptor else None
    linear._tma_descriptor = lambda weight, block_n, block_k: stub
    try:
        linear._project(torch.zeros((m, k), dtype=torch.bfloat16), torch.zeros((n, k), dtype=torch.bfloat16),
                        config, split_ok=True, strict=strict)
    finally:
        linear._tma_gemm, linear._trans_gemm, linear._tma_descriptor = real
    return tma.calls, trans.calls


failures = 0
for m, (n, k) in itertools.product((16, 32, 5), ((6144, 2560), (2560, 4096), (19456, 2560), (2560, 9728), (151936, 2560))):
    configs = [config for config in linear._candidates(m, n, k) if config[0] == "tma"]
    assert len(configs) == 1, configs
    config = configs[0]
    # Fail-safe: no descriptor (this is a CPU) -> the trans kernel; strict (validation) -> an exception.
    tma_calls, trans_calls = record(m, n, k, config, descriptor=False)
    assert not tma_calls and len(trans_calls) == 1, "fallback must be trans"
    try:
        record(m, n, k, config, descriptor=False, strict=True)
        raise AssertionError("strict mode must raise without a descriptor")
    except RuntimeError:
        pass
    (grid, args, kwargs), = record(m, n, k, config, descriptor=True)[0]
    constants = {key: value for key, value in kwargs.items() if key not in ("num_warps", "num_stages")}
    assert grid[0] * constants["BLOCK_N"] == n and constants["SPLITS"] * constants["CHUNK"] == k
    assert constants["CHUNK"] % constants["BLOCK_K"] == 0
    try:
        ptr = "*bf16" if constants["SPLITS"] == 1 else "*fp32"
        out = compile_kernel(linear._tma_gemm, {"x_ptr": "*bf16", "desc_ptr": "*i8", "out_ptr": ptr}, constants,
                             num_warps=kwargs["num_warps"], num_stages=kwargs["num_stages"])
        copies = out.asm["ttgir"].count("async_tma_copy_global_to_local")
        bulk = out.asm["ptx"].count("cp.async.bulk.tensor.2d.shared")
        stores = out.asm["ttgir"].count("async_tma_copy_local_to_global")
        weight_loads = out.asm["ttgir"].count("tt.load")
        print("compiled", m, n, k, config, grid, {key: constants[key] for key in ("SPLITS", "CHUNK", "EVEN_M", "WIDE")},
              f"ttgir tma copies={copies} tma stores={stores} tt.load={weight_loads} ptx bulk-tensor={bulk}", flush=True)
        if not copies or not bulk or stores:
            failures += 1
            print("NO TMA LOWERING", m, n, k)
    except Exception as error:  # noqa: BLE001
        failures += 1
        print("COMPILE FAILED", m, n, k, config, repr(error)[:300])
print("failures", failures)
