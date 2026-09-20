import re
from offline_compile import compile_kernel
from kernels.linear import _gemv, _skinny_gemm
for words, ptr in ((False, "*bf16"), (True, "*i64")):
    a = compile_kernel(_gemv, {"x_ptr": "*bf16", "weight_ptr": ptr, "out_ptr": "*bf16"}, {"N": 19456, "K": 2560, "SPLITS": 1, "CHUNK": 2560, "BLOCK_N": 8, "BLOCK_K": 512, "WORDS": words}, num_warps=4)
    b = compile_kernel(_skinny_gemm, {"x_ptr": "*bf16", "weight_ptr": ptr, "out_ptr": "*fp32"}, {"M": 16, "N": 2560, "K": 9728, "SPLITS": 8, "CHUNK": 1280, "BLOCK_N": 64, "BLOCK_K": 128, "WORDS": words}, num_warps=4, num_stages=2)
    print("words" if words else "plain", "gemv loads/thread:", len(re.findall("ld.global", a.asm["llir"])), "gemm loads/thread:", len(re.findall("ld.global", b.asm["llir"])))
