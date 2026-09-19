# Engine experiments

## 2026-09-19 — static cache, graphed decode, grouped SDPA, fused RMSNorm

Baseline source: commit `e35c206` (unchanged starter).

Candidate 1, commit `0d92f17`, passed the official H100 run
`6ae6665b-46fc-4733-bfc2-02175fd0542a` and ranked at **528.596 tokens/s**.
The complete report and logs are in `agent/results/`. Peak reported GPU memory
was 13.943 GiB. All correctness, latency, memory and timing-stability gates
passed. The score covers six hidden cases; their individual metrics are not
published. Public measurements (five samples each):

| Case | Tokens/s | Speedup vs native | TTFT/native | TPOT/native |
| --- | ---: | ---: | ---: | ---: |
| public-0 | 132.324 | 2.403x | 0.846 | 0.400 |
| public-1 | 274.978 | 1.847x | 0.800 | 0.461 |
| public-2 | 1792.752 | 2.717x | 0.784 | 0.340 |

Candidate implementation:

- Native causal SDPA prefill through the loaded Qwen decoder layers.
- Preallocated BF16 per-layer KV buffers, populated in place.
- One CUDA graph for a single decode step, prepared during shape warmup.
- Device-side absolute position and mask; each generation overwrites its prompt
  and resets the graph inputs. Unused cache capacity is masked.
- Decode maps each group of query heads to SDPA query rows against one KV head,
  avoiding the reference's fourfold KV duplication. It remains dense attention.
- Bundled Triton RMSNorm installed at all norm sites, preserving the reference's
  BF16 cast before multiplying by the learned weight.
- Native RoPE values precomputed for the shape; final normalization and LM head
  evaluate only the last prefill token.

Read-only Claude review suggested the grouped-SDPA optimization. Its other
capture warnings are addressed by device-side masking, a one-element position
tensor, same-stream eager warmup, and synchronous token-list reads.
The implementation review found no concrete cache/graph defect. It prompted
stricter norm and tiny-model checks, A-to-B-to-A shape coverage, a Triton version
check, and moving yields outside inference-mode contexts.

Local verification:

```sh
python3 -m unittest discover -s tests -v
./bin/dryft validate engine
git diff --check
```

The protocol tests use a fake GPU backend and cover output count, batch order,
EOS, zero/one output, shape replacement, and reset across successive prompts.
They do not establish GPU correctness.

GPU verification (requires the pinned runtime, no downloads by the script):

```sh
python agent/verify_gpu.py
python agent/verify_gpu.py --model-path /path/to/provided/checkpoint
```

This checks norm parity, grouped attention versus repeated KV heads, multiple
graph shapes, repeated prompts with poisoned finite stale cache slots, and the
2-logit teacher-forced rule on each emitted prefix. The full checkpoint mode
uses the three public workload shapes. Platform timing remains authoritative.

Current docs supplied by the user supersede the starter's public-run workflow:
only official runs remain, and six hidden workloads determine geometric-mean
tokens/second. A push to the connected default branch creates a submission and
may start its official run. Prepare and validate before requesting that run.
Some starter CLI/client commands still describe retired direct uploads/public
runs; use the connected repository workflow.

Inspect TTFT/native and TPOT/native (both <= 1.10), every correctness result,
sample spread (<= 25%), peak memory (<= 90%), and load/warmup time. The main
remaining risks are masked-SDPA backend selection, full-capacity decode work on
long continuations, and fused-kernel numerical differences. Do not add more
arithmetic changes until a result isolates this candidate's behavior.

## Candidate 2 — packed projections and fused SwiGLU

Starting from candidate 1, concatenate each layer's Q/K/V projection weights
and gate/up projection weights during loading. Each group now uses one native
BF16 linear operation. Keep head normalization, RoPE, residual additions, cache
layout and graph replay unchanged. Replace SiLU plus multiplication with one
Triton kernel that rounds the SiLU result to BF16 before multiplying by up.

Hypothesis: fewer small GEMMs and intermediate kernel launches improve decode;
the combined projections may also improve prefill. Numerical risk: larger GEMM
output dimensions may select a different cuBLAS reduction algorithm; fused SiLU
must retain its intermediate cast. The next official run must establish whether
the overall score improves. Candidate 1 is the known passing fallback.

Result: commit `a339a7f` passed official run
`b4e2dd03-7f39-43a4-a73f-f172128e55fb`, ranked **541.648 tokens/s**, up **2.47%**.
Peak GPU memory was 15.125 GiB. Public TPOT improved from 7.203/9.767/7.798 ms
to 6.741/9.558/7.558 ms. Public TTFT increased to 24.987/166.157/155.906 ms,
while still passing paired native latency gates. Native timings also changed
substantially between runs, so this is a measured score improvement rather than
a controlled attribution of every timing difference to packing alone.

## Candidate 3 — fuse decode Q/K normalization, RoPE and cache writes

Starting from candidate 2, fuse only the single-token decode path. One Triton
program per batch row and Q/K head computes the per-head norm and RoPE. Q heads
write the contiguous query buffer; K heads write the current K cache slot and
copy the corresponding V into its cache slot. Preserve FP32 norm accumulation,
BF16 normalization/gain boundaries, and separate BF16 rounding of both RoPE
products before their sum. Prefill remains the candidate-2 implementation.

Hypothesis: replace the two head-norm kernels, separate rotary operations and
two cache index-copy kernels with one launch per layer, improving TPOT without
changing prefill. Risks: head/stride/cache indexing and compiler cast behavior.
Candidate 2 is the known passing fallback.

Result: commit `1799644` passed official run
`59987926-4597-4a46-aee2-54a2036afec5`, ranked **624.197 tokens/s**, up **15.24%**.
Public TPOT was 5.836/8.327/6.245 ms; TTFT was 22.129/163.551/152.504 ms.
All correctness, latency, memory and stability gates passed. Peak GPU memory
was 15.123 GiB. Native timings again varied, so use the score as observed
performance rather than claiming a controlled causal speedup.

## Candidate 4 — fuse residual additions into RMSNorm

Starting from candidate 3, combine each attention residual with its following
post-attention norm. Defer each MLP residual until the next layer's input norm,
or the final last-token norm. The kernel stores the BF16 residual sum and its
normalized output; it rounds the sum before FP32 variance accumulation, and
rounds normalized activations before multiplying by the learned gain.

This removes 72 separate residual-add launches per decode step. The same
row-independent fusion applies to prefill. Read-only Claude review found no
blocking residual-order, aliasing, cast or graph issues. CPU protocol tests and
archive checks cannot prove GPU parity; the next official run supplies that
evidence. Candidate 3 remains the measured fallback.

Pinned implementation references:

- [Qwen3 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)
- [SDPA adapter 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/integrations/sdpa_attention.py)
- [CUDA graph API 2.5.1](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/graphs.py)
