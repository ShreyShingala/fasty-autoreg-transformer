# Continue — Dryft Qwen3 engine

## State (2026-09-19, ~09:55 UTC)

Leaderboard best: **945.538 tokens/s** (run `273743f8`, commit `344ca6d`), #2
behind Segfault (993.1 at 08:46 UTC). That 945.5 was a favourable draw: the same
engine re-ran at 933.1. The reproducible level since candidate 13 (`e20537c`,
934.6) is about 933 +/- 5. Candidate 19 (`baa5ae1`, lean in-situ refinement)
was pushed; read its result from `GET /api/v1/runs` first.

Session progression: c7 873.0 -> c8 885.9 (prefill Q/K fusion, last-query) ->
c9 926.0 (async streaming) -> c10/11 925.5 (no gain) -> c12 932.0 -> c13 934.6
-> c14 945.5 (variance) -> c15 934.2 -> c16 930.2 -> repro 933.1 -> c17 932.6
-> c18 canceled (15-minute run limit).

## Next action

If c19 is >= c13 within noise, keep it; otherwise return the engine to
`e20537c`'s behaviour (known good, 8.5-minute runs). Then the honest options:

1. Re-run the best engine a few times: identical code varies ~1.3% and the
   leaderboard keeps the best run.
2. Get a real H100 (user has a Modal CLI; spending money needs their OK) to
   profile the decode step. Blind tuning is exhausted: projections, attention
   layouts, launch widths, lossless 12-bit weights and word-sized loads all
   failed to beat the incumbent in the captured step.
3. With a GPU: a hand-written CUDA GEMV via NVRTC is the most plausible +5-10%;
   exact speculation remains high-risk under the 25% spread gate.

## Hard facts

- Whole run is capped at **15 minutes** (c18 canceled); nine fresh processes
  each pay load + warmup. Keep runs under ~11 minutes.
- Only official runs; engine stdout hidden; find runs by `commitSha`.
- TPOT is GPU step time. Batch-one step 3.93 ms: ~3.2 ms projections, ~0.7 ms
  small kernels. Prefill is within ~10% of cuBLAS + Flash.
- Offline H100-target Triton compilation works in `fasty-cpucheck:3.1.0`
  (`agent/local_cpu/`); the Triton interpreter does not.

## Do not

- Reset the dirty tree, reveal `.env`, or start duplicate runs.
- Claim the 2,000 target; it is unmet and not reachable by kernel tuning.
- Touch `agent/ATTACK_PLAN.md` (another actor's untracked file).
