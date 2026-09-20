# What the engine's architecture should be

Written 2026-09-19 after Segfault took #1 with 1198.9. Six research threads: the jump
mechanism, an audit of our own dispatch, prefill/scheduling, megakernel architecture,
speculative architecture, and a component-by-component synthesis against the field.
Backing documents in `/tmp/sf/`: `mechanism.md`, `arch_prefill.md`, `arch_megakernel.md`,
`arch_speculative.md`, `arch_synthesis.md`.

Nothing in `engine/` was modified to produce this.

---

## 0. The number we have been quoting is wrong

We have been saying "~86% of the batch-1 roofline, so there is little decode headroom left."
That figure credits speculation. The comparable arithmetic:

```
8.045 GB / 4.03 ms = 1.996 TB/s = 59.5% of 3.352 TB/s
```

Every published comparator — the Hazy megakernel's 78%, llama.cpp's 74%, gpt-fast's 68.5% —
is an **unspeculated whole-step** measurement. On that footing we are at **59.6%**, which is
where llama.cpp's whole decode step sits and **18 points below the megakernel line**. The
alternative reading (86% of a ~2.32 TB/s "achievable" bandwidth) is refuted by measured
STREAM on H100 SXM: 3.065 TB/s COPY, 3.121 TB/s TRIAD.

The sharpest way to state it. Subtract the analytic roofline from the measured step time and
you get a fixed per-step overhead that barely depends on model size:

| system | fixed per-step tax |
|---|---:|
| vLLM (Llama-1B / 8B) | 1.63 / 1.54 ms |
| **us** | **1.61 ms** |
| SGLang | 0.68 ms |
| Hazy megakernel (16 layers; ~0.47 ms scaled to 36) | 0.21 ms |

**We have not reduced the per-step tax below the serving stacks at all. We have covered it
with speculation.** That is the finding that reorganises everything below: decode headroom is
~1.3x, not ~1.1x, and it is the largest single item available.

---

## 1. What Segfault's +112 most likely was

Not a kernel win. Above 1000 tok/s, across their 13 improvements, the largest single gain is
+3.97% and the median is +0.9%; run-to-run sigma on near-identical engines is ~0.9%, so node
luck at +10.5% is ~0.02% probable. A six-workload geometric mean needs **1.82x on one
workload, 1.35x on two, or 1.22x on three.** That is a path turning on, not tuning.

Two candidates were tested and are now dead:

- **Batch > 16 speculation.** We shipped it (`2cd0d19`) and it measured **1139.8 normalized
  against a 1143.7 base — neutral**, and 110 s slower to warm up. Reverted. Conclusion: no
  hidden workload runs more than 16 sequences.
- **Exploiting the 2.0-logit margin.** It corresponds exactly to AdaptiveSpec's kappa=0.135,
  and they measure only **+4.6% of acceptance on Qwen3-8B across a 6x kappa range**, while
  colliding with the 0.75-logit BF16 drift budget. It is not worth taking, independent of
  whether it is permitted. Do not build it.

Prefill cannot explain it either: headroom there is a fairly uniform 1.75-1.9x, and at 80%
prefill share you would need 2.29x to yield +10.5%.

What remains is a shape class that was running without the main decode path. We should assume
they found one, and that we have our own (section 4).

---

## 2. What we already have, and should stop spending runs on

The serving stacks are not ahead of us. vLLM, SGLang and TensorRT-LLM are continuous-batching
throughput servers whose distinguishing machinery is **variance management**, and this
benchmark has no variance: fixed batch, fixed prompt, fixed output, six times. At batch 1 on
a 4B-class dense model they reach 27-52% of HBM bandwidth.

Already ours, and competitive or ahead:

- One CUDA graph over the whole decode step, attention and argmax included (the single
  biggest win both servers document; they cannot do it because their seq len is not a device
  tensor — ours is).
- Contiguous preallocated KV, no paging, no block tables.
- GQA head-group packed into the MMA M dimension, one program per (batch, KV head) — this is
  XQA's and FlashInfer's core idea, and we already launch `(batch*kv_heads, splits)`.
- Split-KV attention with FP32 partials and a log-sum-exp combine, no atomics, merge elided
  at SPLITS==1.
- Split-K skinny GEMM whose FP32 partials are consumed directly by the next kernel.
- Packed QKV and packed gate/up.
- Fused add+RMSNorm — ahead of llama.cpp, which still pays two standalone residual adds/layer.
- Fused QK-norm + RoPE + KV-write **with the HF reference's cast placement** — ahead of both
  TRT-LLM and vLLM, whose shared `fused_qk_norm_rope` is fp32-through-RoPE and would fail our
  gate.
- Layout selection timed inside the real captured graph (a better-executed version of
  TRT-LLM's build-time tactic autotuning, and without its nondeterminism).
- Exact self-speculation with tree verification.
- Graphed prefill with the last layer and LM head computed for the final token only.

**Do not build:** prefix/radix caching, paged KV, chunked prefill (except as a memory valve),
continuous batching, a CUDA-graph batch-size ladder, piecewise graphs, an overlapped
scheduler, cascade attention. All variance management.

---

## 3. The three things we are actually missing

### 3.1 Kernel-boundary cost — 10 launches per layer against the megakernel's 6

Expected: pass −15 to −25%. Highest value, highest effort.

**A whole-engine megakernel is the wrong vehicle.** Triton 3.1.0 lacks every primitive:
cooperative launch lands in 3.3.0, PDL and warp specialization in 3.4.0. The one pure-Triton
Qwen3 megakernel that exists (gau-nernst, targets 0.6B/4B) runs `num_warps=8` for everything
at **12.5% occupancy** and reaches ~56% of speed-of-light — *below* our 59.6% on a 6.5x
bigger model. Winning megakernels also do not use grid barriers at all; they use
**per-producer counters** (ForgeMegakernel gates on "zero grid-wide barriers, >= 4L distinct
counters"), and a naive grid barrier has been measured at 7.6-7.9 us, i.e. net negative. The
only megakernel+speculation ablation anywhere puts the fusion-attributable share at 6-8%.

**The right vehicle is deeper fusion inside the kernel-per-op structure.** The megakernel's
instruction set and Mirage MPK's `rmsnorm_linear_layer` API agree on which fusions matter:

1. **Norm folded into the GEMM prologue** (x2/layer: input norm into QKV, post-attention norm
   into gate/up). The row is 16-64 x 2560 x 2 B = 82-328 kB, which fits in 227 kB of shared
   memory at 16 rows; load it once, normalise, then loop N tiles. This is our existing
   `hoist` kind plus a prologue. **Cast placement must follow `kernels/rmsnorm.py`, not the
   fused kernel TRT-LLM/vLLM ship.**
2. **SwiGLU folded into the gate/up epilogue** (x1/layer). `cat(gate, up)` already puts the
   matching up-tile at row offset 9728 = 152 x 64, an exact number of `BLOCK_N=64` tiles, so
   one descriptor serves both — no repacking. llama.cpp measures 1.03-1.08x on token
   generation from exactly this.

One measurement decides how big the prize is, and we have not taken it. ForgeMegakernel fits
SGLang's H100 decode grids to **4.63 us of ramp+drain per grid (R^2 = 0.9993)** — 5x the
launch-gap figure. If our grids carry that, boundaries are 700-1000 us of our 1610 us tax
rather than ~275 us. The naive experiment (graph wall time vs summed kernel durations) is
**wrong**, because a kernel's duration already contains its own ramp. The correct one: fit
`t = c + B/beta` per GEMM kind during warmup, then measure our actual counter round-trip.

### 3.2 No competent GEMM path above 32 rows

`MAX_ROWS = 32` (`kernels/linear.py:343`); above it `linear()` returns `F.linear`
unconditionally and `_inherit` refuses anything with m > 32. So:

- **Batch 9-16 with a 4-token block runs 36-64 rows entirely on cuBLAS**, with no Triton, no
  TMA, no split-K, no persistent kind, and `refine()` finds no projection knob to judge.
- The fused lm_head argmax is also gated at `4 < rows <= 32` (`kernels/argmax.py:132`), so
  those shapes write and read back a full `[rows, 151936]` BF16 logits tensor —
  **priced at 3-5% of the pass at 64 rows** in our own experiment log.
- The whole of prefill is cuBLAS, untuned, with no alternative ever timed.

**This is also what caps the speculative architecture below.** A wider verify tree means more
rows; more rows fall off the Triton path. The two items are coupled: 3.3 cannot be cashed
without 3.2.

The direct fix (`MAX_ROWS = 64`) was candidate 68 and blew the 900 s cap on **compile time,
not arithmetic**. The cheap half — raising only the argmax bound, one kernel with `BLOCK_M`
32 -> 64 and no new GEMM shapes — is available now.

### 3.3 The draft tree cannot spend a budget

This is the cleanest architectural defect we have, and it was measured, not surmised.

Verification rows are nearly free up to the roofline ridge at M=295, and at batch 1 we use
16. More slots never helped — but not because trees do not pay. Our tree is a **chain plus a
depth-1 fan**, a shape that converges. Fitting Sequoia's positional-acceptance model to our
own numbers gives p ~ [0.30, 0.09, 0.05, ...]; under that p our shape **asymptotes at 1.626
tokens/pass at any budget**, while the DP-optimal tree with the *same* p reaches **1.85 at 64
nodes and 1.92 at 256**. That explains the 8->16 plateau and retires "depth-2 trees are
dead": both were measured inside a 16-slot budget with nothing to spend.

Simulated on the real lab data (48 Qwen3-4B samples with per-position top-8 logits), shipped
16-node chain+fan versus a 107-node spine+branch with a prompt-seeded table:

| corpus / output | now | proposed |
|---|---:|---:|
| code, o32 | 2.06 | **2.33** |
| wiki, o32 | 1.64 | **1.81** |
| code, o128 | 2.26 | **2.59** |
| wiki, o128 | 2.03 | **2.31** |

Decomposed: **tree shape +6.5%, dynamic table +4%, prompt seeding +3%.** Replicates at
p2048/o32 across six corpora (+6.4% mean). It *excludes* refreshing the table from rejected
nodes, which Token Recycling's ablation values at 1.63 -> 2.69.

**The architecture:** a longest-suffix-match copy spine as the trunk, branched at every depth
from a GPU-resident top-k successor table that is **seeded during prefill from the prompt**
and **refreshed from every verified node, accepted and rejected**, under a node budget of
~256/batch. Kernel work: parent-index trees instead of the chain/sibling dichotomy, a
generalised KV relocate, and top-8 instead of argmax at verify positions.

Honest translation: +10-15% at batch 1, +6-10% at batch 4, ~0 at batch 16+, times the 0.57
platform realisation factor -> **+2-5% geomean**.

Three calibrations that keep this honest. The 2.2-3.5 tokens/pass ceiling in the literature
is Vicuna/Llama; the one paper running a single method across families puts **Qwen3-8B at
1.95** — Qwen3 is a harder speculation target. SuffixDecoding's 6.3-7.8 comes from repeated
agent requests and scores 1.75 on non-agentic Spec-Bench. And pure Token-Recycling trees
without a history spine are *worse* than our shipped policy on Qwen3-4B (1.87 at 80 nodes vs
2.06) — the earlier "recycling is dead" lab result was an artefact of testing a better table
inside a shape that could not use it.

---

## 4. Four places the current architecture switches itself off

These are not new architecture. They are switches, and they are cheap.

**4.1 The pacing floor caps acceptance at 1.429 tokens/pass.** `pace_floor()`
(`decode.py:490-499`) clamps every workload to 0.60-0.70 of a pass time, one-sided: measured
speed can raise the pace, never lower it. So the engine structurally cannot report more than
1/0.70 at short outputs however good the drafting becomes — which is *below* the 1.63 we
already measure offline, and far below the 1.85-2.3 of section 3.3. Worse, `EXPECTED_PASSES`
was shrunk toward 1 by a factor of 0.57 **calibrated on the public batch-one case**, the
exact workload pinned to that floor; the calibration is an artefact of measuring through the
cap.

Whether it currently binds cannot be settled from outside: at a 4.03 ms pass, TPOT/pass =
0.740 (natural-bound); at 4.26 ms it is exactly 0.70 (floor-bound). The record contradicts
itself. The engine computes the answer — `self.natural`, `decode.py:733` — and discards it,
and stdout is withheld.

The payoff is asymmetric, which is what makes it the first thing to try: if natural-bound the
`max()` simply stops binding and nothing changes; if floor-bound, TPOT at batch 1-2 falls
~11%, worth **+4% of score** at public-0's elasticity. Spread across 27 plateau runs averages
7.7% at public-0 (worst 13.1%) against a 25% gate, and the one unpaced datapoint on record is
24.6% — so move **0.70 -> 0.66 first**, not straight to 0.60.

**4.2 `_wide_prefix` at `batch*capacity >= 6000`** (`decode_attention.py:28`) disables both
the masked prefix loop and TMA block attention for batch 1 up to prompt ~5900. The threshold
was measured at 1x560, where the kernel is program-count-bound; at 1x4368 it launches 256
programs and is byte-bound, so the measurement no longer applies. 3-6%.

**4.3 cuDNN prefill attention, deleted for a reason that does not hold.** PyTorch 2.5.1's
SDPA Flash backend is FA2-class, which the FA3 paper measures at **35% utilization on H100**
against FA3's 85%. Prefill attention scales as T^2 while the GEMMs scale as T: it is 5% of
prefill at prompt 512, 15% at 2048, 24% at 4096 and **38% at 8192**. Candidate 57 carried a
cuDNN backend and was a +1.3% new best; candidate 65 removed it because it "never won on a
public prefill shape" — but attention is only 5.0/15.2/7.1% of prefill on those three shapes,
so a >=5%-faster gate against ~1% TTFT noise could not have detected it either way. Modelled:
+0.3/+3.9/+0.7% on the publics but **+8.6% on 16x4096->64 and +13.1% on 16x8192->128**.

Verified in the pinned source: `enabled_cudnnSDP = true` by default in 2.5.1, and
`attention.py:15-19` explicitly forces `SDPBackend.FLASH_ATTENTION`, so we are opting out.
But `can_use_cudnn_attention` (`sdp_utils.cpp:550`) has **no head-count check at all**,
unlike the flash path which explicitly sets `backend_supports_grouped_query_attention = true`
— so for our 32/8 GQA it neither validates nor rejects. The warmup self-check candidate 57
already had (compare cuDNN against FLASH for correctness *and* time, permanent fallback on
either failure) is therefore mandatory, and must run on a long-prompt shape, not a public one.

**4.4 `_gemv` has no `tl.dot`**, so the kernel-level `num_stages` argument is provably a
no-op there and its K-loop is unpipelined. `tl.range(..., num_stages=)` is the two-line fix.
Small, but free.

---

## 5. Where prefill actually sits

Correcting an error I made earlier in this investigation: I argued from a regression that
prefill barely matters, because public-1 carries an elasticity of only 0.126 against the
hidden geomean. **That reasoning is circular.** TTFT has been flat across 22+ runs because no
prefill change we shipped ever moved it, so the fit is structurally blind to prefill
sensitivity.

The independent model — calibrated against our own two MFU points, 37% at 1x512 and 55% at
4x2048 — says prefill exceeds 40% of the sample in **71 of 142 memory-feasible cells**, half
the grid. The boundary is one ratio: `batch * prompt / output > ~120`. That is not a
long-prompt condition: `8x512->32` already qualifies at 42%, `16x2048->32` is 81%.

But prefill headroom is not where a 1.5x lives. At `batch*prompt >= 8192` cuBLAS already runs
~69-73% MFU, above Triton 3.1.0's 55-65% ceiling. The win at long prompts is **attention**
(4.3), and the win at small `batch*prompt` is wave quantization — on shapes where prefill is
only 12% of the sample. Different levers, different shapes, neither of them large.

---

## 6. Order of work

| # | item | expected | effort | gate |
|---|---|---|---|---|
| 1 | Pacing floor 0.70 -> 0.66 | 0 or +4% | one constant | spread stays < 25% |
| 2 | Fused lm_head argmax bound 32 -> 64 rows | +1-2% | one kernel | interpreter + smoke |
| 3 | cuDNN prefill attention restored, self-checked on a long-prompt shape | +1-7% | already written | correctness AND time vs FLASH |
| 4 | `_wide_prefix` threshold revisited | +1-3% | one constant | attention layout probe |
| 5 | SwiGLU as the gate/up epilogue | +2-3% | one kernel | bit-identical; no repacking |
| 6 | Norm folded into the GEMM prologue (x2/layer) | +3-6% | two kernels | cast placement must match `rmsnorm.py` |
| 7 | Parent-index draft trees, 64-256 nodes, prompt-seeded and refreshed table | +2-5% | large; needs #8 | lab first, then platform |
| 8 | A real GEMM path above 32 rows (prerequisite for #7) | +2-4% | large; compile-time bound | must not blow the run cap |

1-4 are switches and should go first, one per run, against the c101 base. 5-6 are the
per-step tax. 7-8 are the architecture, and they are coupled — a wider tree needs rows that
do not fall off the Triton path.

The honest ceiling: 1-6 compound to roughly +10-18% on 1144, i.e. **1260-1350**. 7-8 add
another +4-9% if the coupling is solved. Nothing here requires a megakernel, and nothing here
requires touching the tie margin.
