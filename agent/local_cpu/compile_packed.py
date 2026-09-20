import sys
from offline_compile import compile_kernel
from kernels.packed import _packed_gemv, _packed_gemm
for words, plane in ((False, "*u8"), (True, "*i32")):
    ptrs = {"x_ptr": "*bf16", "sm_ptr": plane, "ex_ptr": plane, "ecol_ptr": "*i16", "eval_ptr": "*bf16", "base_ptr": "*u8"}
    compile_kernel(_packed_gemv, {**ptrs, "out_ptr": "*bf16"}, {"N": 19456, "K": 2560, "E": 8, "BLOCK_N": 8, "BLOCK_K": 512, "BLOCK_E": 8, "WORDS": words}, num_warps=4)
    compile_kernel(_packed_gemv, {**ptrs, "out_ptr": "*bf16"}, {"N": 151936, "K": 2560, "E": 12, "BLOCK_N": 16, "BLOCK_K": 256, "BLOCK_E": 16, "WORDS": words}, num_warps=8)
    compile_kernel(_packed_gemm, {**ptrs, "out_ptr": "*fp32"}, {"M": 16, "N": 2560, "K": 9728, "E": 8, "SPLITS": 8, "CHUNK": 1280, "BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_E": 8, "WORDS": words}, num_warps=4, num_stages=2)
    compile_kernel(_packed_gemm, {**ptrs, "out_ptr": "*bf16"}, {"M": 4, "N": 151936, "K": 2560, "E": 12, "SPLITS": 1, "CHUNK": 2560, "BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_E": 16, "WORDS": words}, num_warps=4, num_stages=2)
print("all packed kernels compile for cuda:90")
