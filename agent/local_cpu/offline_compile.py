"""Compile engine Triton kernels for an H100 target without a GPU (type/semantic check only)."""
import sys, time
sys.path.insert(0, "/work/engine")
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget

TARGET = GPUTarget("cuda", 90, 32)

def compile_kernel(fn, signature, constants, **options):
    names = fn.arg_names
    sig = {names.index(k) if False else k: v for k, v in signature.items()}
    ordered = sorted(signature.items(), key=lambda item: names.index(item[0]))
    src = ASTSource(fn=fn, signature={names.index(k): v for k, v in ordered},
                    constants={names.index(k): v for k, v in constants.items()})
    t = time.time()
    out = triton.compile(src, target=TARGET, options=options)
    print(f"compiled {fn.__name__}: stages={list(out.asm.keys())} in {time.time()-t:.1f}s", flush=True)
    return out

if __name__ == "__main__":
    from kernels.swiglu import _swiglu
    compile_kernel(_swiglu, {"packed": "*bf16", "output": "*bf16"}, {"WIDTH": 9728, "BLOCK": 1024}, num_warps=4)
