# Continue — Dryft Qwen3 engine

## State (2026-09-19, ~17:50 UTC)

**Leaderboard #1: 1129.7 tokens/s** (commit `c096f57`, candidate 57; whole run
815 s against the 900 s cap). Silver Bullet 1104.0 (fork
`sivakovivan/silver-transformer`, tracks our main, consented to idea sharing;
their one idea - refine each block size before comparing - is in c60), dryfter
1087.3 (`john-jpet/fast-transformer`), zip 1059.4. User target: 1200.
Progress today: c48 1095.1 -> c53 1097.7 (GEMM tiles, keep_native) -> c54
1115.0 (two-stage Triton argmax + trimmed budgets) -> c57 1129.7 (stale-guess
sibling + cuDNN prefill option + frozen GC).
In the queue: c58 `663a895` (32 MiB cuBLAS workspace), c60 `b36c802`
(refine-before-compare + hoisted-x GEMM candidate, refine budget 8 s).
RULES OF THE ROAD: keep one run measuring + one queued; record run duration
(`finishedAt - startedAt`) with every score - c52 was canceled at 917 s; every
warmup second costs six. The harness hides engine stdout on purpose (hidden
shapes could be encoded in text): do NOT build side channels (e.g. through
peak memory) - refused once already.
LOCAL GATES before every push (all exist, ~3 min total):
`python3 -m unittest discover -s tests`; `./bin/dryft validate engine`;
`agent/local_cpu/interp/all.sh` (every kernel executed on CPU by a patched
Triton interpreter); `~/.cache/fasty-lab/venv/bin/python
agent/local_cpu/smoke_engine.py` (whole engine, real host control flow, 4-layer
real model vs HF greedy); the offline cuda:90 compile script of any new kernel.
PLAN + research: `agent/NEXT_PLAN.md`, 15 reports in `~/.cache/fasty-lab/plan/`,
paste-able research prompt `agent/RESEARCH_PROMPT.md`.
Key findings: at batch 1 the median sample sits on the pacing floor (0.70 x
pass time) so pass-TIME cuts convert 1:1; history-copy drafting is at its
ceiling (a perfect source selector would save only 3-5% of passes); dead in the
lab today: depth-2 trees, logit re-ranking, hidden-state match selection.
Parked on worktree branches: speculation for batches 17-64
(`worktree-agent-a8f491703758a9453`, ~0 expected gain), fused QK-norm/RoPE/KV +
attention kernel (`worktree-agent-ad7f188c606e1cd22`, bit-exact but compiles
25-35x slower than the two kernels it replaces).
Known bad: cuBLASLt preference, forcing 64-row blocks, ranked-draft kernel
(c49), paired gate/up block kernel (c50).

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
