# Continue — Dryft Qwen3 engine

## State (2026-09-19, ~15:45 UTC)

**Leaderboard #1: 1075.684 tokens/s** (commit `b0e2d76`, candidate 36); teammate
fork "dryfter" 1072.5 (github.com/john-jpet/fast-transformer tracks our main;
the user wants its ideas ported: single-pass block attention and the prefill
gate/up GEMM are in), Segfault 1013.9. User target: 1200. The level since
candidate 29 is about 1065 +/- 10 (run noise); only steps of 2%+ are readable.
The loop runs as the `autoresearch` skill. Collect results without blocking:
`cd agent/tools && python3 collect_runs.py` (rebuilds `agent/results.tsv`).
Queue at this moment, in order: c42 `0f40fa8` (consumers read split-GEMM
partials, no merge launches), c43 `093634e` (two-context successor table +
single-pass attention option; still contains the cuBLASLt and 64-row trials),
c44 `45146de` (= c43 without those two trials), c47 `d10d105` (adds the ported
prefill GEMM, block size measured at warmup, trimmed warmup budgets). Keep the
queue at <= 4 runs; `./bin/dryft cancel <run_id>` drops superseded ones.
A `git stash` holds GEMM occupancy variants (maxnreg / more warps).
Subagents still working when this was written: lab draft-policy loop
(`~/.cache/fasty-lab/REPORT_DRAFTS.md`), skinny-GEMM variant designer
(`~/.cache/fasty-lab/gemm_variants/`).

## What produced the jump from 933

Exact self-speculative decoding (`engine/speculate.py`, `engine/kernels/spec.py`,
block kernels in `kernels/decode_attention.py` and `kernels/qk_rope.py`):
drafts copied from each row's own history (3/2/1-token suffix match) or a
model-derived successor table; one graphed verify pass per block; per-row
positions; rows never pass their last requested token; tokens released no
faster than `PACE` x pass time to bound the 25% spread gate.
c20 948 (batch 1) -> c22 985 (batched) -> c23 1003 (fused bookkeeping) ->
c25 1042 (1-token matches) -> c27 1051 (tree drafts) -> c28 1061 (32-row GEMM)
-> c29 1065 (adaptive pacing). Losers: c26 long chains / 24-row cuBLAS blocks,
c30 32-row blocks at batch 2. Dead offline: token recycling, model-view drafts,
second-level alternatives, lag-based row dealing, frequency votes.

## Tools that now exist

- **Local draft lab** `~/.cache/fasty-lab` (not in the repo): the pinned model on
  MPS, greedy continuations, successor top-8 table, simulators. Draft policies
  can be ranked offline in seconds; the ranking transferred to the platform.
- `agent/local_cpu/`: offline `cuda:90` Triton compilation, CPU emulations of
  every speculation kernel formula (`check_tree.py`, `check_block_attention.py`,
  `check_spec_kernels.py`, `test_speculate.py`, `test_spec_queue.py`).
- Reviews: Claude subagents and `codex exec -s read-only` both work well.

## Known risks

- A rare `incorrect_output` occurred once on the plain batch-16 path (c22 first
  run; identical engine passed on rerun). Failed runs do not replace the best.
- 15-minute whole-run cap; keep warmup lean.
- `PACE` (0.75) binds at batch 1; lowering it trades speed for spread risk.

## Do not

- Reset the dirty tree, reveal `.env`, start duplicate runs, or touch
  `agent/ATTACK_PLAN.md` / `agent/CLAUDE_SYSTEM_PROMPT.md` (untracked, not ours).
