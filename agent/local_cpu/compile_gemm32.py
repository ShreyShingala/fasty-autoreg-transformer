from offline_compile import compile_kernel
from kernels.linear import _skinny_gemm
for m, bm in ((16, 16), (24, 32), (32, 32)):
    compile_kernel(_skinny_gemm, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*fp32"}, {"M": m, "N": 19456, "K": 2560, "SPLITS": 2, "CHUNK": 1280, "BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": bm}, num_warps=4, num_stages=2)
print("gemm up to 32 rows compiles for cuda:90")
