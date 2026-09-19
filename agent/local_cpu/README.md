# CPU-only checks (no CUDA on this Mac)

Image: `fasty-cpucheck:3.1.0` (see `Dockerfile`; Torch 2.5.1 CPU, Triton 3.1.0).
Mount only `engine/` and this directory:

```sh
docker run --rm --platform linux/amd64 -v "$PWD/engine":/work/engine:ro \
  -v "$PWD/agent/local_cpu":/scratch:ro -e PYTHONPATH=/work/engine:/scratch \
  fasty-cpucheck:3.1.0 python /scratch/compile_packed.py
```

- `offline_compile.py`: `triton.compile` with an explicit `GPUTarget("cuda", 90, 32)`.
  Catches Triton type/semantic errors for the H100 target through `cubin`.
  Signature entries must be in argument order (the helper sorts them).
- `compile_packed.py`, `test_pack.py`: packed-weight kernels and codec.
- `check_qk_offsets.py`: flat-offset algebra of the fused Q/K kernel.

`TRITON_INTERPRET=1` does not work in this emulated AMD64 container: even a
trivial store returns zeros. Do not rely on it. None of this proves GPU
execution, numerics or speed; official runs remain the only H100 evidence.

`compile_packed.py`, `count_loads.py` and `test_pack.py` exercise the archived
lossless 12-bit experiment (`agent/archive/packed_lossless_12bit.py`), which is
no longer in `engine/`; mount it as `kernels/packed.py` to rerun them.
`compile_words.py` targets the word-load kernels of commit `072d3e1`.
