import re
from offline_compile import compile_kernel
from kernels.packed import _packed_gemv
from kernels.linear import _gemv
def loads(out): 
    ll = out.asm["llir"]; return {k: len(re.findall(k, ll)) for k in ("ld.global.u8", "ld.global.u32", "ld.global.i32", "ld.global.b16", "ld.global")}
plain = compile_kernel(_gemv, {"x_ptr": "*bf16", "weight_ptr": "*bf16", "out_ptr": "*bf16"}, {"N": 19456, "K": 2560, "SPLITS": 1, "CHUNK": 2560, "BLOCK_N": 8, "BLOCK_K": 512}, num_warps=4)
print("plain ", loads(plain))
for words, plane in ((False, "*u8"), (True, "*i32")):
    ptrs = {"x_ptr": "*bf16", "sm_ptr": plane, "ex_ptr": plane, "ecol_ptr": "*i16", "eval_ptr": "*bf16", "base_ptr": "*u8", "out_ptr": "*bf16"}
    out = compile_kernel(_packed_gemv, ptrs, {"N": 19456, "K": 2560, "E": 8, "BLOCK_N": 8, "BLOCK_K": 512, "BLOCK_E": 8, "WORDS": words}, num_warps=4)
    print("words" if words else "bytes", loads(out), "llir lines", len(out.asm["llir"].split("\n")))
