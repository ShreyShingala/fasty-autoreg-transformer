# Continue — Dryft Qwen3 engine

## State (2026-09-19, ~14:40 UTC)

**Leaderboard #1: 1065.476 tokens/s** (commit `8a7b9f5`, candidate 29); Segfault
1013.0. User target: 1200. The loop is run as the `autoresearch` skill
(`.claude/skills/autoresearch`, also installed in `~/.claude/skills`); the log is
`agent/results.tsv` (rebuild with `python3 agent/tools/make_results_tsv.py`).
Queued on the platform at this moment, in order: c32 `ff9307f` (blocks within 16
rows), c33 `f5adf3b` (verify-graph tile refinement), c34 `622d4fd` (pace floor
0.70), c35 `f3e6c80` (chain depth from match length), c36 `b0e2d76` (warmup never
sets the pace). Watch with `agent/tools/watch_run.py <sha>`. A stash holds the
untried cuBLASLt trial (`git stash list`).

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
