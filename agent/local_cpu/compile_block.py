from offline_compile import compile_kernel
from kernels.qk_rope import _qk_rope_cache
from kernels.decode_attention import _decode_partials, _block_partials, _block_merge
sig = {"packed": "*bf16", "q_weight": "*bf16", "k_weight": "*bf16", "cos": "*bf16", "sin": "*bf16", "position": "*i64", "query": "*bf16", "keys": "*bf16", "values": "*bf16"}
for tokens, prefill, rows in ((1, False, False), (5, False, True), (4, False, True), (512, True, False)):
    compile_kernel(_qk_rope_cache, sig, {"Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": 549, "Q_EPS": 1e-6, "K_EPS": 1e-6, "TOKENS": tokens, "PREFILL": prefill, "BLOCK": 128, "ROWS": rows}, num_warps=1 if prefill else 4)
ptrs = {"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "position_ptr": "*i64", "partial_ptr": "*fp32", "stats_ptr": "*fp32"}
compile_kernel(_decode_partials, ptrs, {"GROUPS": 4, "DIM": 128, "CAPACITY": 549, "SPLITS": 18, "CHUNK": 31, "SCALE": 128 ** -0.5, "BLOCK_M": 16, "BLOCK_N": 32}, num_warps=4, num_stages=2)
for tokens, block_m in ((5, 32), (4, 16), (3, 16)):
    compile_kernel(_block_partials, ptrs, {"TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": 549, "SPLITS": 18, "CHUNK": 31, "SCALE": 128 ** -0.5, "BLOCK_M": block_m, "BLOCK_N": 32}, num_warps=4, num_stages=2)
    compile_kernel(_block_merge, {"partial_ptr": "*fp32", "stats_ptr": "*fp32", "out_ptr": "*bf16"}, {"TOKENS": tokens, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "SPLITS": 18, "BLOCK_S": 32}, num_warps=4)
print("block-mode kernels compile for cuda:90")
