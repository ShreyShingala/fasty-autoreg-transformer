"""Does widening a tile GEMM from 4 to 8 warps buy latency hiding or cost residency?

Candidate 103 showed the GEMMs are limited by how many CTAs are concurrently
streaming, not by bytes moved. More warps per CTA is the other way to raise the
number of loads in flight. This prints registers, shared memory and the
resulting CTAs/SM (H100: 65536 registers and 227 KB of shared memory per SM)
for 4 and 8 warps, plus the total warps resident, which is what actually hides
latency.
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

REGS_PER_SM, SMEM_PER_SM, SMS = 65536, 227 * 1024, 132
SHAPES = {"qkv": (6144, 2560), "o": (2560, 4096), "gate_up": (19456, 2560), "down": (2560, 9728)}
SIG = {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*fp32"}
CUOBJDUMP = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")


class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


def regs(out):
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as handle:
        handle.write(out.asm["cubin"])
        name = handle.name
    try:
        text = subprocess.run([CUOBJDUMP, "-res-usage", name], capture_output=True, text=True).stdout
    finally:
        os.unlink(name)
    found = re.search(r"REG:(\d+)", text)
    spill = re.search(r"STORE:(\d+)", text)
    return int(found.group(1)), int(spill.group(1)) if spill else 0


print(f"{'shape':8s} {'m':>3s} {'warps':>5s} {'grid':>5s} {'regs':>5s} {'spill':>5s} {'smem':>6s} "
      f"{'CTA/SM':>6s} {'slots':>5s} {'waves':>5s} {'warps/SM':>8s}")
for (label, (n, k)), m in itertools.product(SHAPES.items(), (16, 32)):
    config = linear._candidates(m, n, k)[0]          # the shipped "gemm" kind
    for warps in (4, 8):
        rec = Recorder()
        saved = linear._skinny_gemm
        linear._skinny_gemm = rec
        try:
            linear._project(torch.zeros((m, k), dtype=torch.bfloat16),
                            torch.zeros((n, k), dtype=torch.bfloat16), config, split_ok=True)
        finally:
            linear._skinny_gemm = saved
        grid, _, kwargs = rec.calls[0]
        constants = {key: value for key, value in kwargs.items() if key not in ("num_warps", "num_stages")}
        try:
            out = compile_kernel(saved, SIG, constants, num_warps=warps, num_stages=2)
            register, spill = regs(out)
            threads = 32 * warps
            ctas = max(1, min(REGS_PER_SM // max(1, register * threads), SMEM_PER_SM // max(1, out.metadata.shared), 32))
            total = grid[0] * (grid[1] if len(grid) > 1 else 1)
            print(f"{label:8s} {m:3d} {warps:5d} {total:5d} {register:5d} {spill:5d} {out.metadata.shared:6d} "
                  f"{ctas:6d} {ctas*SMS:5d} {-(-total // (ctas*SMS)):5d} {ctas*warps:8d}")
        except Exception as error:  # noqa: BLE001
            print(f"{label:8s} {m:3d} {warps:5d} FAILED {repr(error)[:90]}")
