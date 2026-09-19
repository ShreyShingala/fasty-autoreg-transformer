"""Lossless 12-bit storage of BF16 projection weights for memory-bound decode.

This is a storage layout, not quantization: every weight is reconstructed to
its exact BF16 bit pattern before use, so the products and FP32 accumulation
are those of the plain BF16 kernels in ``kernels.linear``.

A BF16 value is sign(1) | exponent(8) | mantissa(7). Trained weights occupy a
narrow band of exponents, so one byte holds sign+mantissa and a 4-bit code
holds ``exponent - base`` for a per-matrix ``base`` (codes 1..15). Code 0 with
a zero byte is +0.0. The rare weights outside the window ("exceptions": tiny
values, outliers, denormals, -0.0) are stored as +0.0 in the planes and kept
exactly, as BF16, in a small dense per-row side table that the same kernel
adds in FP32 before the single BF16 rounding. A matrix whose rows need more
than ``MAX_EXCEPTIONS`` entries is simply not packed.

``pack`` verifies bit-exact reconstruction of every weight before returning.
"""

import torch
import triton
import triton.language as tl

MAX_EXCEPTIONS = 16


class Packed:
    __slots__ = ("sm", "ex", "base", "ecol", "eval", "width", "shape")

    def __init__(self, sm, ex, base, ecol, evalues, width, shape):
        self.sm, self.ex, self.base = sm, ex, base
        self.ecol, self.eval, self.width, self.shape = ecol, evalues, width, shape


def _fields(weight):
    bits = weight.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = (bits >> 7) & 0xFF
    sign_mantissa = ((bits >> 8) & 0x80) | (bits & 0x7F)
    return bits, exponent, sign_mantissa


def _chunks(weight):
    """Row blocks of about 2**24 weights, bounding load-time temporaries."""
    return weight.split(max(1, (1 << 24) // weight.shape[1]), dim=0)


def best_base(weight):
    """The base whose window [base+1, base+15] covers the most weights."""
    histogram = torch.zeros(256, dtype=torch.int64, device=weight.device)
    for block in _chunks(weight):
        histogram += torch.bincount(_fields(block)[1].reshape(-1), minlength=256)
    histogram[0] = 0  # zeros and denormals never use a window code
    cumulative = torch.cat((histogram.new_zeros(1), histogram.cumsum(0)))
    covered = cumulative[16:257] - cumulative[1:242]  # base = 0..240
    return int(covered.argmax().item())


def exception_counts(weight, base):
    counts = []
    for block in _chunks(weight):
        bits, exponent, _ = _fields(block)
        code = exponent - base
        counts.append(((bits != 0) & ((code < 1) | (code > 15))).sum(dim=1))
    return torch.cat(counts)


def _restore(sm, ex, base, ecol, evalues):
    rows, columns = sm.shape
    nibbles = ex.to(torch.int32)
    code = torch.stack((nibbles & 15, nibbles >> 4), dim=-1).reshape(rows, columns)
    byte = sm.to(torch.int32)
    exponent = torch.where(code > 0, code + base, torch.zeros_like(code))
    bits = ((byte & 0x80) << 8) | (exponent << 7) | (byte & 0x7F)
    bits = torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16)
    restored = bits.view(torch.bfloat16).clone()
    real = evalues.view(torch.int16) != 0
    row_index = torch.arange(rows, device=restored.device)[:, None].expand_as(ecol)
    restored[row_index[real], ecol[real].to(torch.int64)] = evalues[real]
    return restored


def unpack(packed):
    """Pure-Torch reconstruction, used only to verify a packing."""
    return _restore(packed.sm, packed.ex, packed.base, packed.ecol, packed.eval)


def _pack_rows(weight, base, width):
    rows = weight.shape[0]
    bits, exponent, sign_mantissa = _fields(weight)
    code = exponent - base
    inside = (code >= 1) & (code <= 15)
    exception = (~inside) & (bits != 0)
    counts = exception.sum(dim=1)
    if int(counts.max().item()) > width:
        return None
    code = torch.where(inside, code, torch.zeros_like(code))
    sm = torch.where(inside, sign_mantissa, torch.zeros_like(sign_mantissa)).to(torch.uint8)
    ex = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    ecol = torch.zeros((rows, width), dtype=torch.int16, device=weight.device)
    evalues = torch.zeros((rows, width), dtype=torch.bfloat16, device=weight.device)
    where = exception.nonzero()
    if where.shape[0]:
        row, column = where[:, 0], where[:, 1]
        first = torch.cumsum(counts, 0) - counts
        slot = torch.arange(where.shape[0], device=weight.device) - first[row]
        ecol[row, slot] = column.to(torch.int16)
        evalues[row, slot] = weight[row, column]
    # Every weight must come back bit for bit, or the matrix stays unpacked.
    restored = _restore(sm, ex, base, ecol, evalues)
    if not torch.equal(restored.view(torch.int16), weight.contiguous().view(torch.int16)):
        return None
    return sm, ex, ecol, evalues


def pack(weight, base, width):
    """Pack one contiguous BF16 [N, K] matrix, or return None if it does not fit."""
    rows, columns = weight.shape
    if weight.dtype != torch.bfloat16 or columns % 2 or columns > 32767 or not 1 <= width <= MAX_EXCEPTIONS:
        return None
    parts = []
    for block in _chunks(weight):
        part = _pack_rows(block, base, width)
        if part is None:
            return None
        parts.append(part)
    sm, ex, ecol, evalues = (torch.cat(column).contiguous() for column in zip(*parts))
    return Packed(sm, ex, base, ecol, evalues, width, (rows, columns))


@triton.jit
def _decode32(sm, code, base):
    # BF16 and FP32 share sign and exponent; the 7 mantissa bits are the top
    # of FP32's 23. Code 0 always carries a zero byte, hence +0.0.
    byte = sm.to(tl.uint32)
    exponent = tl.where(code > 0, code + base, 0).to(tl.uint32)
    bits = ((byte & 0x80) << 24) | (exponent << 23) | ((byte & 0x7F) << 16)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _decode16(sm, code, base):
    byte = sm.to(tl.uint32)
    exponent = tl.where(code > 0, code + base, 0).to(tl.uint32)
    bits = ((byte & 0x80) << 8) | (exponent << 7) | (byte & 0x7F)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=["base"])
def _packed_gemv(
    x_ptr, sm_ptr, ex_ptr, ecol_ptr, eval_ptr, out_ptr, base,
    N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_E: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    live = rows < N
    columns = tl.arange(0, BLOCK_K)
    pairs = tl.arange(0, BLOCK_K // 2)
    acc = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
    for start in range(0, K, BLOCK_K):
        k = start + columns
        pair = start // 2 + pairs
        x = tl.load(x_ptr + k, k < K, other=0).to(tl.float32)
        sm = tl.load(
            sm_ptr + rows[:, None] * K + k[None, :],
            live[:, None] & (k[None, :] < K), other=0,
        )
        nibbles = tl.load(
            ex_ptr + rows[:, None] * (K // 2) + pair[None, :],
            live[:, None] & (pair[None, :] < K // 2), other=0,
        ).to(tl.int32)
        code = tl.interleave(nibbles & 15, nibbles >> 4)
        acc += _decode32(sm, code, base) * x[None, :]
    result = tl.sum(acc, axis=1)
    slots = tl.arange(0, BLOCK_E)
    listed = live[:, None] & (slots[None, :] < E)
    column = tl.load(ecol_ptr + rows[:, None] * E + slots[None, :], listed, other=0).to(tl.int64)
    exact = tl.load(eval_ptr + rows[:, None] * E + slots[None, :], listed, other=0).to(tl.float32)
    x_exact = tl.load(x_ptr + column, listed, other=0).to(tl.float32)
    result += tl.sum(exact * x_exact, axis=1)
    tl.store(out_ptr + rows, result, live)


@triton.jit(do_not_specialize=["base"])
def _packed_gemm(
    x_ptr, sm_ptr, ex_ptr, ecol_ptr, eval_ptr, out_ptr, base,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
    SPLITS: tl.constexpr, CHUNK: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_E: tl.constexpr,
):
    rows = tl.arange(0, 16)
    columns = tl.program_id(0).to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    live = columns < N
    reduction = tl.arange(0, BLOCK_K)
    pairs = tl.arange(0, BLOCK_K // 2)
    acc = tl.zeros((16, BLOCK_N), tl.float32)
    for start in range(split * CHUNK, (split + 1) * CHUNK, BLOCK_K):
        k = start + reduction
        pair = start // 2 + pairs
        x = tl.load(
            x_ptr + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K), other=0,
        )
        sm = tl.load(
            sm_ptr + columns[:, None] * K + k[None, :],
            live[:, None] & (k[None, :] < K), other=0,
        )
        nibbles = tl.load(
            ex_ptr + columns[:, None] * (K // 2) + pair[None, :],
            live[:, None] & (pair[None, :] < K // 2), other=0,
        ).to(tl.int32)
        code = tl.interleave(nibbles & 15, nibbles >> 4)
        # The same BF16 tensor-core product with FP32 accumulation as the
        # unpacked kernel: the decoded operands are bit-identical BF16.
        acc = tl.dot(x, tl.trans(_decode16(sm, code, base)), acc)
    if split == 0:
        slots = tl.arange(0, BLOCK_E)
        listed = live[:, None] & (slots[None, :] < E)
        column = tl.load(ecol_ptr + columns[:, None] * E + slots[None, :], listed, other=0).to(tl.int64)
        exact = tl.load(eval_ptr + columns[:, None] * E + slots[None, :], listed, other=0).to(tl.float32)
        x_exact = tl.load(
            x_ptr + rows[:, None, None] * K + column[None, :, :],
            (rows[:, None, None] < M) & listed[None, :, :], other=0,
        ).to(tl.float32)
        acc += tl.sum(x_exact * exact[None, :, :], axis=2)
    tl.store(
        out_ptr + split * M * N + rows[:, None] * N + columns[None, :], acc,
        (rows[:, None] < M) & (columns[None, :] < N),
    )


_REGISTRY = {}


def lookup(weight):
    return _REGISTRY.get(weight.data_ptr())


def clear():
    _REGISTRY.clear()


def register(weights):
    """Pack same-shaped matrices with one exception-table width, or skip them."""
    plans = []
    for weight in weights:
        if weight.dtype != torch.bfloat16 or not weight.is_contiguous() or weight.dim() != 2:
            continue
        base = best_base(weight)
        worst = int(exception_counts(weight, base).max().item())
        if worst <= MAX_EXCEPTIONS:
            plans.append((weight, base, worst))
    if not plans:
        return 0
    width = max(4, triton.cdiv(max(worst for _, _, worst in plans), 4) * 4)
    done = 0
    for weight, base, _ in plans:
        packed = pack(weight, base, width)
        if packed is not None:
            _REGISTRY[weight.data_ptr()] = packed
            done += 1
    return done


def project(x, packed, config, merge):
    """x BF16 [M, K] with M <= 16; ``merge`` reduces FP32 split partials."""
    kind, block_n, block_k, splits, warps = config
    m, k = x.shape
    n = packed.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    block_e = triton.next_power_of_2(packed.width)
    if kind == "pgemv":
        _packed_gemv[(triton.cdiv(n, block_n),)](
            x, packed.sm, packed.ex, packed.ecol, packed.eval, out, packed.base,
            N=n, K=k, E=packed.width, BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_E=block_e,
            num_warps=warps,
        )
        return out
    partial = out if splits == 1 else torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    chunk = triton.cdiv(k, splits * block_k) * block_k
    _packed_gemm[(triton.cdiv(n, block_n), splits)](
        x, packed.sm, packed.ex, packed.ecol, packed.eval, partial, packed.base,
        M=m, N=n, K=k, E=packed.width, SPLITS=splits, CHUNK=chunk,
        BLOCK_N=block_n, BLOCK_K=block_k, BLOCK_E=block_e, num_warps=warps, num_stages=2,
    )
    if splits > 1:
        merge(partial, out, m * n, splits)
    return out


def exact_columns(weight, packed, config, merge, rows):
    """One-hot inputs must return the chosen BF16 columns bit for bit.

    A unit input makes every product exact and every other term +0.0, so any
    error in the byte planes, nibble order, base, or exception table shows.
    """
    n, k = weight.shape
    listed = (packed.eval.view(torch.int16) != 0).nonzero()
    columns = [0, 1, k // 2 - 1, k // 2, k - 2, k - 1]
    columns += packed.ecol[listed[:6, 0], listed[:6, 1]].to(torch.int64).tolist()
    for begin in range(0, len(columns), rows):
        chosen = columns[begin:begin + rows]
        chosen += [chosen[-1]] * (rows - len(chosen))
        x = torch.zeros((rows, k), device=weight.device, dtype=weight.dtype)
        x[torch.arange(rows), chosen] = 1.0
        actual = project(x, packed, config, merge)
        expected = weight[:, chosen].t().contiguous()
        if not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
            return False
    return True
