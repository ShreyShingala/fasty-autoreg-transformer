# Research assignment: exact BF16 Qwen3 inference on H100

You are an independent inference-performance researcher working alongside an
agent that is actively implementing and benchmarking an engine. Your task is
to identify the next highest-value optimizations, challenge the implementation's
assumptions, and propose concrete experiments. Research deeply using primary
sources, pinned source code, and GPU architecture documentation. Do not merely
list generic inference techniques.

## Objective and benchmark

We are competing in the Dryft Kernel Rush challenge. Run the fixed
`Qwen/Qwen3-4B-Instruct-2507` checkpoint, revision
`cdbee75f17c01a7cc42f958dc650907174af0554`, on **one NVIDIA H100 80GB HBM3**.
Maximize the geometric mean of output tokens/second across **six hidden
workloads**. Throughput includes prefill. Three additional public workloads
provide diagnostic measurements:

| Case | Batch | Prompt per sequence | Output per sequence |
| --- | ---: | ---: | ---: |
| public-0 | 1 | 512 | 32 |
| public-1 | 4 | 2048 | 32 |
| public-2 | 16 | 512 | 128 |

There are five measured samples per workload. Each workload uses a fresh engine
process: load, one untimed warmup generation of the same shape, then samples.
Warmup can compile kernels, select implementations, allocate buffers, and
capture CUDA graphs. Reuse weights and shape-specific state, but never prompt
content across samples. Prompts are fresh and unknown. Do not infer hidden
shapes or exploit the benchmark harness.

Every case must pass:

- Each emitted token must be native Qwen's greedy argmax on OUR emitted prefix,
  or at most **2.0 logits** below that argmax. The judge replays the whole output
  teacher-forced after our process exits. One bad position fails the workload.
- TTFT and TPOT must each be <= **1.10 times native**.
- Timing spread across five samples must be <= **25%**.
- Peak GPU memory <= **90%** of the device.
- Load plus warmup <= **300 seconds**; each sample <= **300 seconds**.
- Yield exactly the requested number of steps: one host list of token IDs per
  step, one per sequence in original batch order. EOS is an ordinary token.

Only official runs remain. They run on the remote platform; we do not have a
local CUDA GPU. Do not claim a kernel is faster or numerically validated from
source inspection alone. Published run output does not reveal hidden shapes.

## Runtime and numerical constraints

Pinned runtime: Python 3.11, CUDA 12.4, PyTorch **2.5.1**, Triton **3.1.0**,
Transformers **4.51.3**, safetensors 0.5.3, tokenizers 0.21.1.

The submission is Python/Triton source only, under 2 MiB compressed and 200 files.
No network, installs, compiled binaries, additional weights, external services,
or downloads inside the engine. The checkpoint is provided at `model_path`.
FlashAttention/FlashInfer/vLLM/SGLang implementations may be research references,
but do not assume their packages or newer APIs exist in this runtime. Explain
what can actually be ported as small Python/Triton source modules.

No quantization, reduced-precision weights or KV cache, cache eviction, sparse or
approximate attention, pruned vocabulary, unverified draft tokens, or changed
model. Exact speculation is legal but must reproduce native verification.
BF16 reordering noise is tolerated; the margin is not permission to approximate.
Native cached decode itself can differ by roughly 0.75 logits from full replay.

**Cast boundaries matter.** Native RMSNorm reduces and normalizes in FP32,
casts the normalized activation to BF16, then multiplies by its BF16 weight.
Native SwiGLU rounds `silu(gate)` to BF16 before multiplying it by `up` and
rounding again. RoPE multiplies and addition also have native tensor rounding
boundaries. A fused FP32 reformulation that rounds only once is not equivalent.
Preserve projection outputs, residual rounding, head ordering, and lowest-index
argmax tie behavior.

## Model architecture

- 36 decoder layers; hidden width **2560**; MLP width **9728**.
- **32 Q heads**, **8 KV heads**, **128 dimensions/head**; four Q heads per KV head.
- Q width is **4096**, not 2560. K and V are each **1024** wide.
- No linear biases; SwiGLU MLP; dense causal attention, no sliding window.
- Q and K receive per-head RMSNorm before RoPE. V does not.
- RMSNorm epsilon `1e-6`; RoPE theta `5_000_000` and absolute positions.
- Vocabulary **151936**; BF16 input embedding and LM-head weights are tied.
- BF16 KV payload: `147456 * batch * sequence_length` bytes across all layers.
- Layer order: input norm -> QKV -> Q/K head norm -> RoPE -> attention -> output
  projection -> residual -> post-attention norm -> gate/up -> SiLU/product ->
  down projection -> residual.

## Repository and current implementation

Repository: `https://github.com/ShreyShingala/fasty-autoreg-transformer`
Local folder, if you share the workspace:
`/Users/shrey/Downloads/Coding/fasty-autoreg-transformer`

Read `QWEN_ENGINE_CONTRACT.md` and `OPTIMIZATION_GUIDE.md` completely, then read
the actual implementation. Their public-run/scoring workflow is older than the
current rules above. Main files:

- `engine/engine.py`: loading and host token stream; only `Engine.__init__` and
  `Engine.generate` are public entry points.
- `engine/decode.py`: fixed KV buffers, native RoPE table, direct layer dispatch,
  warmup, and one CUDA graph per `(batch, prompt_length, output_length)` shape.
- `engine/attention.py`: grouped decode SDPA with no replicated KV heads.
- `engine/layers.py`: packed QKV and packed gate/up projections (candidate 2).
- `engine/kernels/rmsnorm.py`, `engine/kernels/swiglu.py`: fused kernels.
- `agent/EXPERIMENTS.md`: hypotheses and measured results.
- `agent/verify_gpu.py`: numerical checks intended for a GPU runtime; NOT run
  locally. CPU tests cover only the generator contract, not GPU arithmetic.

Baseline starter is commit `e35c206`. **Candidate 1, commit `0d92f17`, passed
an official H100 run and ranked at 528.596 tokens/s**, with peak GPU memory
13.943 GiB. All gates passed. Its implementation:

1. Load native BF16 weights, TF32 disabled. Preserve native matrix products.
2. Replace all hidden and Q/K RMSNorms with the fused Triton kernel.
3. Preallocate KV `[B, 8, prompt+output, 128]` per layer. Prefill copies K/V into
   the prefix but returns only fresh prompt K/V to native causal SDPA.
4. Decode writes K/V with GPU-position `index_copy_`. Fixed-capacity attention
   uses an explicit GPU mask `arange(capacity) <= position`, updated in the graph.
5. For single-token decode, reshape Q `[B,32,1,128]` to `[B,8,4,128]`, treating
   query heads within each KV group as SDPA query rows. Attend directly against
   `[B,8,C,128]` K/V. `is_causal=False`: these four rows are heads at the SAME
   time position, not four successive tokens. This avoids native 4.51.3's
   fourfold K/V materialization. Prefill stays on the native SDPA adapter.
6. Precompute native RoPE cos/sin during warmup. Capture decode forward, argmax,
   next-token copy, and position increment in one CUDA graph. Host `.tolist()`
   and yield happen synchronously outside capture, one step at a time.
7. During prefill, final norm and LM head process only the last hidden token.

Candidate 1 official public measurements (five-sample medians):

| Case | Candidate tok/s | Candidate TTFT ms | Native TTFT ms | Candidate TPOT ms | Native TPOT ms | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| public-0 | 132.324 | 18.962 | 22.405 | 7.203 | 18.027 | 2.403x |
| public-1 | 274.978 | 162.859 | 203.514 | 9.767 | 21.198 | 1.847x |
| public-2 | 1792.752 | 150.978 | 192.456 | 7.798 | 22.926 | 2.717x |

**Candidate 2, commit `a339a7f`, passed at 541.648 tokens/s (+2.47%).** It packs
Q/K/V into one BF16 linear projection with output width 6144, packs gate/up
into one projection with output width 19456, and fuses SwiGLU while retaining
the BF16 SiLU intermediate. Everything else is substantially unchanged.
Packing occurs during load and releases the original separate weight tensors.
Run ID: `b4e2dd03-7f39-43a4-a73f-f172128e55fb`. Public TPOT improved to
6.741/9.558/7.558 ms; TTFT increased to 24.987/166.157/155.906 ms, with every
gate passing. Native times also moved between runs, so avoid attributing every
timing difference to code alone.

**Candidate 3 is being implemented:** decode-only fusion of Q/K head RMSNorm,
RoPE and KV writes in `engine/kernels/qk_rope.py`. Coordinate before duplicating
this work; research the next opportunities and adversarially review this kernel.

## Research questions — prioritize these

1. **Where is the remaining decode time plausibly spent?** Build a quantitative
   lower-bound analysis from weight bytes, KV traffic, GEMM shapes, H100 HBM
   bandwidth, and launch count. Separate measured facts from hypotheses; do not
   invent a profiler trace. Account for the large tied LM head.
2. **Small-batch matrix products:** when can a source-only Triton BF16 GEMV or
   GEMM beat PyTorch/cuBLAS for these exact shapes? Consider B=1,4,16 plus other
   possible batches. Propose tiling, split-K/reduction, coalescing, accumulation
   precision and dispatch. Explain register pressure and why a hand-written
   kernel might lose to cuBLAS. Is packed projection really the right layout?
3. **Decode attention:** compare the current grouped SDPA trick against a custom
   dense split-K attention kernel, including reading a device-side valid length,
   online softmax, full-capacity mask overhead, and GQA reuse. Confirm the
   backends actually available in torch 2.5.1. Do not recommend installing
   FlashInfer or relying on unsupported `enable_gqa` behavior.
4. **Safe fusion:** rank Q/K norm + RoPE + KV write, residual + norm, activation
   epilogues, and LM-head argmax. Spell out every required BF16 cast and tensor
   layout. Estimate attainable savings relative to development/numerical risk.
5. **Prefill:** identify real opportunities without worsening TTFT. Consider
   native SDPA backend selection, packed projections, temporary copies, and
   chunking only if a quantitative benefit is credible.
6. **Exact speculation:** evaluate whether deterministic n-gram/prompt lookup
   or another legal draft can help without extra model weights. Describe exact
   verification, batch acceptance, cache rollback, and streaming latency. State
   the acceptance rate needed to beat the current engine; recommend against it
   if the likely economics are poor.
7. **Adversarial review:** identify concrete correctness or stability risks in
   our current implementation and proposed changes, including state reuse,
   graph capture, shape specialization, peak memory, and numeric drift.

## Required deliverable

Give us an actionable research report, not a broad literature survey:

1. A ranked table of **5–8 specific opportunities** with expected mechanism,
   affected workloads, rough gain range explicitly labeled as an estimate,
   implementation effort, numerical risk, and runtime compatibility.
2. The **best three experiments**, in recommended execution order. For each:
   exact code/module boundary, algorithm or kernel pseudocode, tensor shapes,
   launch strategy, cast boundaries, correctness tests, and go/no-go benchmark
   criteria. Make the experiments separable so a result can be attributed.
3. Primary-source links with pinned versions or commits and the precise claim
   each supports. Distinguish portable ideas from implementations requiring
   Hopper-specific compiled CUDA extensions or newer Triton/PyTorch features.
4. A short section of **ideas to reject or defer**, with concrete reasons.
5. Any uncertainty that needs an actual H100 measurement; ask for the smallest
   useful measurement instead of presenting speculation as a fact.

Prefer a small, measurable improvement we can implement and validate next over
an ambitious redesign with no credible latency or correctness argument.

If you share the repository: work read-only on `engine/`, do not commit or push,
do not start/cancel official runs, do not read `.env` or credentials, and do not
modify the active candidate. Put your report under `agent/research/` or return
it in your response. Coordinate with the implementing agent before any edits.
