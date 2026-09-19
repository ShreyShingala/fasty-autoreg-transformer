import sys
sys.path.insert(0, "/work/engine")
import torch
from kernels.packed import pack, unpack, best_base, exception_counts, MAX_EXCEPTIONS
torch.manual_seed(0)
def kernel_view(p):
    """Mirror _decode32/_decode16 and the kernel's nibble order in Torch; exceptions excluded."""
    rows, cols = p.shape
    ex = p.ex.to(torch.int32)
    lo, hi = ex & 15, ex >> 4
    code = torch.stack((lo, hi), -1).reshape(rows, cols)          # tl.interleave(lo, hi)
    byte = p.sm.to(torch.int64)
    exponent = torch.where(code > 0, code + p.base.to(torch.int32)[:, None], torch.zeros_like(code)).to(torch.int64)
    bits32 = ((byte & 0x80) << 24) | (exponent << 23) | ((byte & 0x7F) << 16)
    bits32 = torch.where(bits32 >= 2**31, bits32 - 2**32, bits32).to(torch.int32)
    return bits32.view(torch.float32)
for n, k, scale in ((64, 256, 0.02), (33, 514, 0.05), (128, 2560, 0.015)):
    w = (torch.randn(n, k) * scale).to(torch.bfloat16)
    w[0, 0] = 0.0; w[0, 1] = -0.0; w[1, 5] = 3.0; w[2, 7] = 1e-30; w[3, 9] = torch.tensor(1e-39).to(torch.bfloat16)  # zero, -0, outlier, tiny, denormal/0
    base = best_base(w); counts = exception_counts(w, base)
    width = max(4, min(MAX_EXCEPTIONS, -(-int(counts.max()) // 4) * 4))
    p = pack(w, base, width)
    assert p is not None, (n, k, int(counts.max()))
    assert torch.equal(unpack(p).view(torch.int16), w.view(torch.int16))
    # kernel-side value = decoded planes + exact exception table contributions
    dense = kernel_view(p).double()
    real = p.eval.view(torch.int16) != 0
    rows_idx = torch.arange(n)[:, None].expand_as(p.ecol)
    assert (dense[rows_idx[real], p.ecol[real].long()] == 0).all(), "exception slots must decode to +0.0 in the planes"
    dense[rows_idx[real], p.ecol[real].long()] = p.eval[real].double()
    assert torch.equal(dense, w.double()), "kernel-side reconstruction differs"
    x = torch.randn(k).to(torch.bfloat16)
    y_ref = (w.float() @ x.float()); y_pk = (dense.float() @ x.float())
    assert torch.equal(y_ref, y_pk)
    bytes_packed = p.sm.numel() + p.ex.numel() + p.ecol.numel() * 2 + p.eval.numel() * 2
    print(f"[{n}x{k}] bases={sorted(set(base.tolist()))[:4]} exceptions total={int(counts.sum())} max/row={int(counts.max())} width={width} bytes={bytes_packed/(n*k*2):.3f} of BF16")
# a row needing too many exceptions must refuse to pack
w = (torch.randn(8, 64) * 0.02).to(torch.bfloat16); w[0, :40] = 1e-30
assert pack(w, best_base(w), MAX_EXCEPTIONS) is None
print("pack codec OK: bit-exact incl. zero, -0.0, outlier, tiny, denormal; oversize rows refused")

# chunked path: force several chunks and the registry
import kernels.packed as P
w = (torch.randn(700, 2560) * 0.02).to(torch.bfloat16)
orig = P._chunks; P._chunks = lambda t: t.split(97, dim=0)
b = best_base(w); c = exception_counts(w, b); pk = pack(w, b, max(4, -(-int(c.max()) // 4) * 4))
assert pk is not None and torch.equal(unpack(pk).view(torch.int16), w.view(torch.int16)) and pk.sm.shape == (700, 2560) and pk.ex.shape == (700, 1280)
P._chunks = orig
ws = [(torch.randn(64, 256) * 0.02).to(torch.bfloat16) for _ in range(3)]
assert P.register(ws) == 3 and all(P.lookup(t) is not None for t in ws) and len({P.lookup(t).width for t in ws}) == 1
print("chunked packing and registry OK")

# word-granular views must expose the same bytes/codes in column order
w = (torch.randn(40, 2560) * 0.02).to(torch.bfloat16)
b = best_base(w); pk = pack(w, b, 4)
words = pk.sm32.to(torch.int64)
sm_from_words = ((words[:, :, None] >> (torch.arange(4) * 8)) & 0xFF).reshape(40, 2560)
assert torch.equal(sm_from_words, pk.sm.to(torch.int64))
codes = pk.ex32.to(torch.int64)
code_from_words = ((codes[:, :, None] >> (torch.arange(8) * 4)) & 15).reshape(40, 2560)
ex = pk.ex.to(torch.int64)
assert torch.equal(code_from_words, torch.stack((ex & 15, ex >> 4), -1).reshape(40, 2560))
print("int32 word views match the byte planes in column order")

# output channels with very different scales: one window per matrix would refuse this
w = torch.randn(256, 2560) * (2.0 ** torch.randint(-14, 2, (256, 1)).float()) * 0.02
w = w.to(torch.bfloat16)
b = best_base(w); c = exception_counts(w, b)
pk = pack(w, b, max(4, -(-int(c.max()) // 4) * 4))
assert pk is not None and torch.equal(unpack(pk).view(torch.int16), w.view(torch.int16))
print(f"mixed-scale rows pack with per-row bases: distinct bases={len(set(b.tolist()))} max exceptions/row={int(c.max())}")
