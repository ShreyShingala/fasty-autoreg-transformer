# Continue — Dryft Qwen3 engine

## State (2026-09-19, ~16:20 UTC)

**Leaderboard #1: 1095.135 tokens/s** (commit `b1ca1cc`, candidate 48; whole run
710 s against the 900 s cap); teammate fork "dryfter" 1087.3
(github.com/john-jpet/fast-transformer follows our main; single-pass block
attention, the prefill gate/up GEMM and the paired block kernel are ported),
Segfault 1013.9. User target: 1200. Run noise is about +/-1%; only steps of 2%+
are readable. The loop runs as the `autoresearch` skill. Collect results without
blocking: `cd agent/tools && python3 collect_runs.py` (rebuilds
`agent/results.tsv`).
QUEUE DISCIPLINE (the user complained about a messy submissions page): at most
two runs queued; hold finished work locally; cancel superseded runs with
`./bin/dryft cancel <run_id>`.
In the queue: c49 `80b0ec4` (c48 + ranked candidate drafts), c50 `662342b`
(+ paired gate/up block kernel as a refine option).
HELD LOCALLY (not pushed), candidate 51 = commits `53063e7`..HEAD on main:
mask-free `exact` / transposed `trans` GEMM tiles with whole-block split counts
(`engine/kernels/gemm.py`), layout inheritance for the second block size tried
at warmup (it used to run cuBLAS-only once the 24 s tuning budget was spent),
refine ordered by per-step weight traffic, refine skips a projection knob the
paired kernel replaces. Push it only after c49/c50 results say which of their
changes stay; if c49 regresses, revert the ranked drafts with a new commit first.
Known bad: cuBLASLt preference, forcing 64-row blocks for every short prompt.
A `git stash` holds older GEMM occupancy variants (maxnreg / more warps; the lab
found maxnreg only spills and num_stages changes nothing).

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
