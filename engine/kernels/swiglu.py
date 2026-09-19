"""SwiGLU from packed gate/up projections, preserving the SiLU BF16 output."""

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu(packed, output, count, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = index // WIDTH
    column = index % WIDTH
    valid = index < count
    gate = tl.load(packed + row * (2 * WIDTH) + column, valid, other=0).to(tl.float32)
    up = tl.load(packed + row * (2 * WIDTH) + WIDTH + column, valid, other=0).to(tl.float32)
    # Native SiLU returns BF16 before the separate gate/up multiplication.
    activated = (gate / (1.0 + tl.exp(-gate))).to(output.dtype.element_ty).to(tl.float32)
    tl.store(output + index, activated * up, valid)


def swiglu(packed):
    """Contiguous BF16 [..., 2*I] gate/up input; new contiguous [..., I] output."""
    width = packed.shape[-1] // 2
    output = torch.empty((*packed.shape[:-1], width), device=packed.device, dtype=packed.dtype)
    count = output.numel()
    _swiglu[(triton.cdiv(count, 1024),)](packed, output, count, WIDTH=width, BLOCK=1024)
    return output
