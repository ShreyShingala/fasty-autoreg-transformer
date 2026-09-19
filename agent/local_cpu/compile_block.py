from offline_compile import compile_kernel
from kernels.qk_rope import _qk_rope_cache
from kernels.decode_attention import _decode_partials
sig = {"packed": "*bf16", "q_weight": "*bf16", "k_weight": "*bf16", "cos": "*bf16", "sin": "*bf16", "position": "*i64", "query": "*bf16", "keys": "*bf16", "values": "*bf16"}
for tokens, prefill in ((1, False), (5, False), (512, True)):
    compile_kernel(_qk_rope_cache, sig, {"Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": 564, "Q_EPS": 1e-6, "K_EPS": 1e-6, "TOKENS": tokens, "PREFILL": prefill, "BLOCK": 128}, num_warps=1 if prefill else 4)
ptrs = {"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "position_ptr": "*i64", "partial_ptr": "*fp32", "stats_ptr": "*fp32"}
for shared in (False, True):
    compile_kernel(_decode_partials, ptrs, {"GROUPS": 4, "DIM": 128, "CAPACITY": 564, "SPLITS": 7, "CHUNK": 81, "SCALE": 128 ** -0.5, "BLOCK_M": 16, "BLOCK_N": 64, "KV_HEADS": 8, "SHARED": shared}, num_warps=4, num_stages=2)
print("block-mode kernels compile for cuda:90")
