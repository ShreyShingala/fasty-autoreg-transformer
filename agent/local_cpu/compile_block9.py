from offline_compile import compile_kernel
from kernels.decode_attention import _block_partials, _block_merge
from kernels.qk_rope import _qk_rope_cache
ptrs = {"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "position_ptr": "*i64", "partial_ptr": "*fp32", "stats_ptr": "*fp32"}
for tokens, block_m in ((9, 64), (8, 32), (3, 16), (2, 16)):
    compile_kernel(_block_partials, ptrs, {"TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": 553, "SPLITS": 18, "CHUNK": 31, "SCALE": 128 ** -0.5, "BLOCK_M": block_m, "BLOCK_N": 32}, num_warps=4, num_stages=2)
print("ok")
