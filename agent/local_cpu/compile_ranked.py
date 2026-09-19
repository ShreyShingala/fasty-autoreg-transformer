from offline_compile import compile_kernel
from kernels.spec import _propose_ranked
sig = {"history": "*i64", "position": "*i64", "successor": "*i64", "weights": "*fp32", "tokens": "*i64", "chains": "*i64", "phases": "*i64"}
for size, T, D in ((551, 16, (5, 8, 13, 14)), (2086, 4, (1, 2, 3, 3)), (662, 2, (1, 1, 1, 1)), (551, 8, (2, 4, 6, 7))):
    block = 1 << (size - 1).bit_length()
    compile_kernel(_propose_ranked, sig, {"SIZE": size, "TOKENS": T, "MAXLEN": 8, "D0": D[0], "D1": D[1], "D2": D[2], "D3": D[3], "HIST": 8, "TOP": 8, "BLOCK": block, "BLOCK_C": 16, "BLOCK_T": 1 << (T - 1).bit_length()}, num_warps=16)
print("ranked propose compiles for cuda:90")
