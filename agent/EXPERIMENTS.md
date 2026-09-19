# Engine experiments

## 2026-09-19 — static cache, graphed decode, grouped SDPA, fused RMSNorm

Baseline source: commit `e35c206` (unchanged starter). No baseline or candidate
H100 result has been obtained in this session. No throughput improvement is
claimed yet.

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
unmeasured risks are masked-SDPA backend selection, full-capacity decode work on
long continuations, and fused-kernel numerical differences. Do not add more
arithmetic changes until a result isolates this candidate's behavior.

Pinned implementation references:

- [Qwen3 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py)
- [SDPA adapter 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/integrations/sdpa_attention.py)
- [CUDA graph API 2.5.1](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/cuda/graphs.py)
