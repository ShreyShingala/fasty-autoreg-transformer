from offline_compile import compile_kernel
from kernels.spec import _propose, _settle
for size, count in ((551, 4), (2086, 3), (70, 1)):
    block = 1 << (size - 1).bit_length()
    compile_kernel(_propose, {"history": "*i64", "position": "*i64", "successor": "*i64", "tokens": "*i64"}, {"SIZE": size, "COUNT": count, "BLOCK": block}, num_warps=4)
    compile_kernel(_settle, {"tokens": "*i64", "greedy": "*i64", "position": "*i64", "limit": "*i64", "history": "*i64", "result": "*i64"}, {"SIZE": size, "TOKENS": count + 1, "BLOCK": 1 << count.bit_length()}, num_warps=1)
print("speculation bookkeeping kernels compile for cuda:90")
