from offline_compile import compile_kernel
from kernels.spec import _propose, _settle, _relocate
for size, drafts, sibs in ((551, 8, 7), (2086, 2, 1), (70, 1, 0), (551, 4, 3)):
    block = 1 << (size - 1).bit_length(); T = 1 + drafts + sibs
    compile_kernel(_propose, {"history": "*i64", "position": "*i64", "successor": "*i64", "tokens": "*i64"}, {"SIZE": size, "DRAFTS": drafts, "SIBLINGS": sibs, "ALTERNATES": min(sibs, 3), "TOP": 8, "BLOCK": block, "BLOCK_S": max(1, 1 << (max(sibs, 1) - 1).bit_length())}, num_warps=4)
    compile_kernel(_settle, {"tokens": "*i64", "greedy": "*i64", "position": "*i64", "limit": "*i64", "history": "*i64", "result": "*i64", "move_from": "*i64", "move_to": "*i64"}, {"SIZE": size, "TOKENS": T, "CHAIN": 1 + drafts, "BLOCK": 1 << (T - 1).bit_length()}, num_warps=1)
compile_kernel(_relocate, {"store": "*bf16", "move_from": "*i64", "move_to": "*i64"}, {"BATCH": 2, "KV_HEADS": 8, "CAPACITY": 553, "DIM": 128, "BLOCK_H": 8}, num_warps=1)
print("speculation bookkeeping kernels compile for cuda:90")
