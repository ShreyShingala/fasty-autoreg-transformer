from offline_compile import compile_kernel
from kernels.argmax import _block_argmax, _first_best
compile_kernel(_block_argmax, {"logits": "*bf16", "best_value": "*fp32", "best_index": "*i64"}, {"VOCAB": 151936, "BLOCKS": 19, "BLOCK": 8192}, num_warps=4)
compile_kernel(_first_best, {"best_value": "*fp32", "best_index": "*i64", "out": "*i64"}, {"BLOCKS": 19, "BLOCK_B": 32}, num_warps=1)
print("argmax kernels compile for cuda:90")
