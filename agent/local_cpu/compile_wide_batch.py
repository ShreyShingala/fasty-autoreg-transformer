# Verify blocks of batches 17-64 (decode.block_candidates): three tokens per row
# up to batch 21, else two; cuBLAS-sized row counts, one or two attention
# intervals. Both sizes are compiled for batches <= 32 (a superset).
from offline_compile import compile_kernel
from kernels.spec import _propose, _settle, _relocate
from kernels.qk_rope import _qk_rope_cache
from kernels.decode_attention import _block_partials, _block_merge

DRAFTS = {2: (1, 1, 1, 1), 3: (1, 2, 2, 2)}
# (batch, prompt, output): capacity = prompt + output + widest candidate.
for batch, prompt, output in ((17, 512, 128), (32, 2048, 32), (64, 2048, 128)):
    widest = 3 if 3 * batch <= 64 else 2
    capacity = prompt + output + widest
    size = capacity + 2
    for T in ((2, 3) if batch <= 32 else (2,)):
        D = DRAFTS[T]
        lanes = T - 1 - min(D)
        compile_kernel(_propose, {"history": "*i64", "position": "*i64", "successor": "*i64", "tokens": "*i64", "chains": "*i64", "phases": "*i64"}, {"SIZE": size, "TOKENS": T, "MAXLEN": 8, "D0": D[0], "D1": D[1], "D2": D[2], "D3": D[3], "LANES": lanes, "ALTERNATES": min(lanes, 3), "TOP": 8, "BLOCK": 1 << (size - 1).bit_length(), "BLOCK_S": max(1, 1 << (max(lanes, 1) - 1).bit_length()), "BLOCK_T": 1 << (T - 1).bit_length()}, num_warps=4)
        compile_kernel(_settle, {"tokens": "*i64", "greedy": "*i64", "position": "*i64", "limit": "*i64", "history": "*i64", "result": "*i64", "move_from": "*i64", "move_to": "*i64", "chains": "*i64"}, {"SIZE": size, "TOKENS": T, "BLOCK": 1 << (T - 1).bit_length()}, num_warps=1)
        for warps in (4, 1, 2):
            compile_kernel(_qk_rope_cache, {"packed": "*bf16", "q_weight": "*bf16", "k_weight": "*bf16", "cos": "*bf16", "sin": "*bf16", "position": "*i64", "query": "*bf16", "keys": "*bf16", "values": "*bf16"}, {"Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": capacity, "Q_EPS": 1e-6, "K_EPS": 1e-6, "TOKENS": T, "PREFILL": False, "BLOCK": 128, "ROWS": True, "COUNT": 1, "SPLITS": 1}, num_warps=warps)
        # block_attention's default layout and the alternatives refine() may try.
        default = min(32, -(-256 // (batch * 8)), -(-capacity // 64))
        layouts = {(64, default)}
        for block_n, splits in ((64, 1), (128, 1), (128, default // 2), (64, default * 2), (128, default)):
            layouts.add((block_n, max(1, min(32, splits, -(-capacity // block_n)))))
        for block_n, splits in sorted(layouts):
            ptrs = {"q_ptr": "*bf16", "k_ptr": "*bf16", "v_ptr": "*bf16", "position_ptr": "*i64", "chain_ptr": "*i64", "partial_ptr": "*fp32", "stats_ptr": "*fp32", "out_ptr": "*bf16"}
            if splits == 1:
                ptrs["partial_ptr"] = ptrs["stats_ptr"] = "*bf16"
            compile_kernel(_block_partials, ptrs, {"TOKENS": T, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "CAPACITY": capacity, "SPLITS": splits, "CHUNK": -(-capacity // splits), "SCALE": 128 ** -0.5, "BLOCK_M": max(16, 1 << (T * 4 - 1).bit_length()), "BLOCK_N": block_n}, num_warps=4, num_stages=2)
            if splits > 1:
                compile_kernel(_block_merge, {"partial_ptr": "*fp32", "stats_ptr": "*fp32", "out_ptr": "*bf16"}, {"TOKENS": T, "GROUPS": 4, "Q_HEADS": 32, "KV_HEADS": 8, "DIM": 128, "SPLITS": splits, "BLOCK_S": 1 << (splits - 1).bit_length()}, num_warps=4)
    compile_kernel(_relocate, {"store": "*bf16", "move_from": "*i64", "move_to": "*i64"}, {"BATCH": batch, "KV_HEADS": 8, "CAPACITY": capacity, "DIM": 128, "BLOCK_H": 8}, num_warps=1)
print("verify-block kernels of batches 17-64 compile for cuda:90")
