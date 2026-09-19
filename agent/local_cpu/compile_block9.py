from offline_compile import compile_kernel
from kernels.decode_attention import _block_partials, _block_merge
ptrs = {"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "position_ptr": "*i64", "partial_ptr": "*fp32", "stats_ptr": "*fp32"}
for tokens, chain, block_m in ((16, 9, 64), (8, 5, 32), (5, 4, 32), (4, 3, 16), (3, 3, 16), (2, 2, 16)):
    compile_kernel(_block_partials, ptrs, {"TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": 560, "SPLITS": 18, "CHUNK": 32, "SCALE": 128 ** -0.5, "BLOCK_M": block_m, "BLOCK_N": 32, "CHAIN": chain}, num_warps=4, num_stages=2)
    compile_kernel(_block_merge, {"partial_ptr": "*fp32", "stats_ptr": "*fp32", "out_ptr": "*bf16"}, {"TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "SPLITS": 18, "BLOCK_S": 32}, num_warps=4)
print("tree block kernels compile for cuda:90")
