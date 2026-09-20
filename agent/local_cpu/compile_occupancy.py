"""Registers, spills and shared memory per CTA for every GEMM kind the verify pass
actually launches, so residency can be computed without a GPU.

H100: 65536 32-bit registers and 227 KB of usable shared memory per SM.
CTAs/SM = min(regs, smem, 32-block) limit; the bytes-in-flight target is >=123
resident CTAs GPU-wide (3.352 TB/s x 600 ns = 2.01 MB at ~16 KB staged per CTA).
"""
import itertools
import os
import re
import subprocess
import tempfile

import torch
import triton

from offline_compile import compile_kernel
from kernels import linear

REGS_PER_SM = 65536
SMEM_PER_SM = 227 * 1024
SMS = 132

SHAPES = {"qkv": (6144, 2560), "o": (2560, 4096), "gate_up": (19456, 2560),
          "down": (2560, 9728), "lm_head": (151936, 2560)}

SIGS = {
    "_skinny_gemm": {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*fp32"},
    "_trans_gemm": {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*fp32"},
    "_hoist_gemm": {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*fp32"},
    "_tma_gemm": {"x_ptr": "*bf16", "desc_ptr": "*i8", "out_ptr": "*fp32"},
    "_tmap_gemm": {"x_ptr": "*bf16", "desc_ptr": "*i8", "out_ptr": "*bf16"},
    "_persist_trans_gemm": {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*bf16"},
}


def _cuobjdump():
    root = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin")
    path = os.path.join(root, "cuobjdump")
    return path if os.path.exists(path) else None


CUOBJDUMP = _cuobjdump()


def resources(out):
    """(registers, stack, spill stores, spill loads) per thread, from the cubin's ELF."""
    if CUOBJDUMP is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as handle:
        handle.write(out.asm["cubin"])
        name = handle.name
    try:
        text = subprocess.run([CUOBJDUMP, "-res-usage", name], capture_output=True, text=True).stdout
    finally:
        os.unlink(name)
    regs = re.search(r"REG:(\d+)", text)
    stack = re.search(r"STACK:(\d+)", text)
    store = re.search(r"STORE:(\d+)", text)
    load = re.search(r"LOAD:(\d+)", text)
    if regs is None:
        return None
    return (int(regs.group(1)), int(stack.group(1)) if stack else 0,
            int(store.group(1)) if store else 0, int(load.group(1)) if load else 0)


class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


def record(m, n, k, config):
    """Run _project with every kernel replaced by a recorder; return (name, grid, kwargs)."""
    saved = {name: getattr(linear, name) for name in SIGS}
    saved["_tma_descriptor"] = linear._tma_descriptor
    recorders = {name: Recorder() for name in SIGS}
    for name, rec in recorders.items():
        setattr(linear, name, rec)
    stub = torch.zeros(linear.TMA_SIZE, dtype=torch.int8)
    linear._tma_descriptor = lambda weight, block_n, block_k: stub
    try:
        linear._project(torch.zeros((m, k), dtype=torch.bfloat16),
                        torch.zeros((n, k), dtype=torch.bfloat16), config, split_ok=True)
    finally:
        for name, value in saved.items():
            setattr(linear, name, value)
    return [(name, rec.calls[0]) for name, rec in recorders.items() if rec.calls]


print(f"{'shape':9s} {'kind':7s} {'m':>3s} {'kernel':20s} {'grid':>6s} {'regs':>5s} "
      f"{'spill':>5s} {'smem':>7s} {'reg/SM':>6s} {'sm/SM':>6s} {'CTA/SM':>6s} {'slots':>6s} {'waves':>5s}")
for (label, (n, k)), m in itertools.product(SHAPES.items(), (16, 32)):
    for config in linear._candidates(m, n, k):
        kind = config[0]
        try:
            for name, (grid, args, kwargs) in record(m, n, k, config):
                constants = {key: value for key, value in kwargs.items()
                             if key not in ("num_warps", "num_stages")}
                out = compile_kernel(linear.__dict__[name], SIGS[name], constants,
                                     num_warps=kwargs["num_warps"], num_stages=kwargs.get("num_stages", 2))
                usage = resources(out)
                if usage is None:
                    print("no cuobjdump"); raise SystemExit(1)
                regs, spills = usage[0], usage[2] + usage[3]
                smem = out.metadata.shared
                threads = 32 * kwargs["num_warps"]
                by_reg = REGS_PER_SM // max(1, regs * threads)
                by_smem = SMEM_PER_SM // max(1, smem)
                ctas = max(1, min(by_reg, by_smem, 32))
                slots = ctas * SMS
                total = grid[0] * (grid[1] if len(grid) > 1 else 1)
                print(f"{label:9s} {kind:7s} {m:3d} {name:20s} {total:6d} {regs:5d} {spills:5d} "
                      f"{smem:7d} {by_reg:6d} {by_smem:6d} {ctas:6d} {slots:6d} "
                      f"{-(-total // slots):5d}")
        except Exception as error:  # noqa: BLE001
            print(f"{label:9s} {kind:7s} {m:3d} FAILED {repr(error)[:110]}")
