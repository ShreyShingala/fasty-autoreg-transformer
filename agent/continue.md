# State - 2026-09-19, late

Leaderboard: **SSS 1144.3 (#1)**, dryfter 1138.7, Silver Bullet 1137.7, then
zip 1071.1. The two teams behind us are our own merged queues; the nearest
real rival is 73 points back.

## All three queues are run queues

`agent/tools/dispatch.sh <mate|silver> <sha> "<message>"`. **New: dryfter's API
token is in the ignored `.env` as `DRYFT_TOKEN_MATE`.** Read their runs with
`DRYFT_TEAM=MATE python3 collect_runs.py` and
`RESULTS=../results_mate python3 report_runs.py`, so a dispatched candidate now
reports per-workload numbers instead of only moving that team's leaderboard
best. Silver Bullet still reads out only through the leaderboard.

In flight: SSS `4fbebc7` (c102, mailbox NumPy view), dryfter `9a68694` (c103,
split-K retune), Silver Bullet `aa472ef` (c104, refine budget share).

## Read a score against the node before believing it

`report_runs.py` prints the control. Candidate 101 - `engine/` byte-identical
to the 1144.3 tree - came back **1053.1** because native's own prefill TTFT was
263.6 / 251.2 ms against the calibration node's 202.3 / 192.0. Anything more
than ~5% off that is measuring the node. The normalizer's 1374.9 for that run
is an extrapolation, not a reading.

## What the last round established

- `num_stages` 2->3 on every tile GEMM: **exactly neutral** (1143.6 vs 1143.7,
  measured on dryfter). The PTX genuinely differs (26/39/51 `cp.async`).
  Deeper pipelining is closed.
- Residency is register-bound, not shared-memory-bound, for the default GEMM:
  `_skinny_gemm` is 64 registers over 128 threads against 20 KB of smem, so
  8 CTAs/SM = **1056 slots**, and no per-layer grid waves. The TMA kinds are
  smem-bound at 5 CTAs/SM = 660. Zero register spills anywhere. Method:
  offline cuda:90 compile, then `cuobjdump -res-usage` on the cubin
  (`agent/local_cpu/compile_occupancy.py`).
- **`lm_head` is the one wave-quantized GEMM**: 2374 CTAs over 1056 slots =
  3 waves, 4-5 on `trans`. Nothing has attacked it.
- Split counts were never tuned - `_choose` searches kind and block_n only.

## Next, in order

1. **Block attention runs on 8 of 132 SMs at batch 1** (`SPLITS == 1` after
   refine). 350 us against a 33 us byte floor; 100-200 us recoverable. Biggest
   single non-GEMM item -> dryfter, it is architectural.
2. `lm_head` wave quantization (3 deep) - persistent/stream-K decomposition.
3. Lead-in replay before the timed group (`decode.py:469-487`): the host's
   first `cudaGraphLaunch` gap sits inside the timed interval, biasing
   `pass_seconds` up, and at batch 1 the median sample sits on
   `0.70 x pass_seconds`, so the bias converts 1:1.
4. Fold QK-norm/RoPE/KV-write into attention (-36 launches).
5. `_embed_rms_norm`'s `n_cols` is a runtime arg, so it emits 64 scalar b16
   loads; as `tl.constexpr` it compiles to 8 `ld.global.v4`. Tiny but free.

Full inventory: `~/.cache/fasty-lab/plan/inventory_pass.md` (30 items with
PTX-derived costs), `scout_round1.md`, `sweep_new.md`.

## The rule

One change per run, each measured against the 1144.3 base (`c758faf`, and
`97d43d2` whose `engine/` is byte-identical). Five stacked unmeasured edits
drifted 1143.7 -> 1109.3 once already. Run-to-run sigma is 1.3%, so a change
under ~3% must be proven offline rather than read off one run.
