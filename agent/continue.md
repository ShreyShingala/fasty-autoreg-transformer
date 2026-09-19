# Continue — Dryft Qwen3 engine

## State (2026-09-19, ~13:55 UTC)

**Leaderboard #1: 1042.440 tokens/s** (commit `8adb604`, run `74855daf`, candidate
25); Segfault 1013.0. The user's next target is 1200. Queued/unread at this
moment: candidate 26 (`607e1f5`, longer chains) and candidate 27 (`80ab1ef`,
tree drafts). Read `GET /api/v1/runs` first. Runs take 8-10 minutes and queue;
do not wait idle for them — build the next candidate meanwhile.

## What produced the jump from 933

Exact self-speculative decoding (`engine/speculate.py`, `engine/kernels/spec.py`,
block kernels in `kernels/decode_attention.py` and `kernels/qk_rope.py`):
drafts copied from each row's own history (3/2/1-token suffix match) or a
model-derived successor table; one graphed verify pass per block; per-row
positions; rows never pass their last requested token; tokens released no
faster than `PACE` x pass time to bound the 25% spread gate.
c20 948 (batch 1) -> c22 985 (batched) -> c23 1003 (fused bookkeeping) ->
c25 1042 (1-token matches).

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
