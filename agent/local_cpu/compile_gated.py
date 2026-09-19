from offline_compile import compile_kernel
from kernels.gated_linear import _paired_projection
for m, opt in ((8192, (32, 64, 32)), (8192, (64, 64, 32)), (512, (32, 64, 32))):
    compile_kernel(_paired_projection, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*bf16"}, {"M": m, "I": 9728, "K": 2560, "BM": opt[0], "BN": opt[1], "BK": opt[2]}, num_warps=4, num_stages=3)
print("paired gate/up projection compiles for cuda:90")
