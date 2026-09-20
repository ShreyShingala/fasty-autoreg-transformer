# State - 2026-09-20 early

Leaderboard: **Segfault 1198.9**, SSS 1144.3, 0xDeadBeaf 1142.9, dryfter
1138.7, Silver Bullet 1137.7. We are #2 and need **+4.8%**. Segfault went
1064.4 -> 1176.4 -> 1198.9 in about an hour; their repository is private and
no public trace of it exists.

## Four run queues, all readable

Tokens live in the ignored `.env`: `DRYFT_TOKEN` (SSS), `DRYFT_TOKEN_MATE`
(dryfter), `DRYFT_TOKEN_SILVER`, `DRYFT_TOKEN_DEAD` (0xDeadBeaf). Read any of
them with `DRYFT_TEAM=<MATE|SILVER|DEAD> python3 collect_runs.py` and
`RESULTS=../results_<lower> python3 report_runs.py`. Dispatch with
`agent/tools/dispatch.sh <mate|silver|dead> <sha> "<msg>"`; SSS is
`git push origin main`. dryfter's own team also pushes to their queue, so check
it before dispatching there.

## Two measurement facts that change how to work

**Paired draws of the same tree agree to ~0.1% normalized.** Candidate 106 drew
1139.8 and 1140.9; candidate 104 drew 1121.5 and 1122.1; the base drew 1143.7
and 1143.6. The 1.33% figure was raw-score spread dominated by node speed. So
**a 0.5% win is readable in one run** and the old "needs 3%" bar is retired.
Always read the normalized column, and discard any run whose node is more than
~5% off.

**A graph node costs 1.0-1.3 us.** Measured directly: 300 empty kernels added
to the verify pass moved TPOT +7.7% / +8.5% / +6.3% on the three public shapes.
So all 331 nodes are ~9% of the pass and fusing the ~144 fusable small kernels
is bounded at **~3.9%**, matching the independent design review's ~2.4%
realistic estimate. A megakernel is not worth it: a grid barrier costs about
what a node costs.

## The central obstacle to the fusion track

Split partials are ~6.4 MB a layer and **L2-resident on a 50 MB L2**. That is
why candidate 103 cost 4.3%: cutting split counts removed no HBM traffic at
all, only concurrent CTAs. And almost every epilogue fusion buys itself by
forcing `SPLITS = 1`, which is exactly where the CTAs come from. Any fusion
proposal must state what happens to the CTA count, or it is not costed.

Note also that decode **already** fuses SwiGLU in the sense that matters: the
split GEMM hands FP32 partials straight to `_swiglu` through `merged.Split`,
so no `[rows, 19456]` BF16 tensor is ever materialised. The remaining prize is
only the 36 launches, ~40 us, ~1.0%.

## Measured and closed

| change | normalized | verdict |
| --- | ---: | --- |
| base `c758faf` / `7928148` | 1143.7 / 1143.6 | the bar |
| c106 speculate above 16 sequences | 1139.8 / 1140.9 | neutral; **no hidden workload is above 16** |
| c102 mailbox NumPy view | 1122.4 | discard |
| c104 refine budget share | 1121.5 / 1122.1 | discard |
| c105 lead-in replay | 1114.4 | discard, and confounded (groups 5->4 raises `min`) |
| c103 split-K target 512->128 | 1094.6 | **-4.3%** |
| `num_stages` 2->3 | 1143.6 | exactly neutral |

Also closed: 8-warp tile GEMMs (same 32 resident warps, half the slots);
persistent `lm_head` (caps programs at the tile count - qkv would run 96 CTAs
against 1056 slots); rebasing the parked fused attention `85b43bb`; the whole
draft side (hindsight oracle bound only +3.1%); larger trees at batch 1 (c26,
c30, d14ca21); quantisation, PDL, weight compression, cache eviction.

**Three separate perturbations of warmup or host measurement each lost ~2%.**
The engine is tuned to its current warmup timing; prefer kernel and arithmetic
changes that leave the tuning loop alone.

## In flight

SSS `0dbe286` (fused lm_head argmax on by default), 0xDeadBeaf `f8b10df` (same
tree, second draw), Silver Bullet `854158d` (constexpr embedding width +
incremental sibling mask).

## Next

1. Read `~/.cache/fasty-lab/plan/restructure.md` before building any fusion.
2. The SwiGLU epilogue is ranked highest of the unbuilt items but **halves
   gate_up's CTAs, 608 -> 304**, which is the candidate-103 trap; price that
   before building, not after.
3. Warmup economy: `torch/cuda/graphs.py:57-59` runs
   `synchronize(); gc.collect(); empty_cache()` on **every** capture, and
   `triton/runtime/build.py:21-48` forks one blocking gcc per launcher. No
   score directly, but runs have reached 804 s against a cap that has already
   killed two.
4. We still do not know what Segfault did. The batch-cap hypothesis is
   disproved and fusion is bounded at ~3.9%.
