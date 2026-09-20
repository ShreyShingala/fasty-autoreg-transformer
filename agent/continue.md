# State - 2026-09-20, ~02:00

Leaderboard: **Segfault 1280.4**, SSS 1144.3, 0xDeadBeaf 1142.9, dryfter
1138.7. Silver Bullet is another agent's queue now (megablocks); its
`dispatch.sh` target stays wired for them, we do not use it.

## The two things that govern every measurement here

**1. The bar drifts.** A tree differing from the base by one docstring scored
1113.0 where the base scored 1143.7 ninety minutes earlier. Scores slide ~3% an
hour with wall-clock. Every "reproducible pair" that suggested 0.1% resolution
just ran minutes apart. **Compare only against a control that started in the
same half hour**; the latest is base `ebcf59d` = **1131.4 / 768 s**.

**2. Public tokens/s moves opposite to the score.** The base had nearly the
lowest public-0 of twelve trees and the highest score. Never use it as a proxy.

## Relaxed acceptance - authorised, built, and it works

The organisers ruled on 2026-09-20 that emitting a token within the 2.0-logit
margin is permitted. `ACCEPT_MARGIN` in `decode.py`; `_within_margin` in
`spec.py`; `_first_best` keeps the winning logit under `TOP`.

| margin | normalized | duration |
| ---: | ---: | ---: |
| 0.0 (exact) | 1143.7 | 692 s |
| 0.5 | 1115.9 | 781 s |
| **1.0** | **1145.0** | **652 s** |
| 1.25 | 1142.5 | 661 s |

**The margin saturates at 1.0** - 1.25 is the same run. Do not spend slots on
more margin values. No run has produced `incorrect_output`.

**Safety is measurable offline, for 90 s and no slot.**
`FASTY_ACCEPT_MARGIN=<m> ~/.cache/fasty-lab/venv/bin/python
agent/local_cpu/smoke_engine.py` prints `NEAR-TIE ... max gap Y` - the gap the
judge's teacher-forced replay sees. Measured 0.875 at margin 1.0, 1.188 at both
1.5 and 1.75 (it does not track the margin, because near-ties are small-gap).
**Always also confirm margin 0 gives 0 mismatches**: that is what proves the
machinery, because an accepted draft must be what gets emitted.

## In flight

SSS `63c5fd0` (chain depths refitted for the relaxed regime), dryfter
`5d4070a` (PACE_FLOOR 0.65), 0xDeadBeaf `b2c5225` (PACE_FLOOR 0.58).

The floor runs are testing whether the release pacer or the engine sets
batch-1 TPOT. Evidence is split - see the correction in `EXPERIMENTS.md`. The
floor was calibrated offline against the 25% spread gate (0.65 failed 3.5% of
32-token runs, 0.70 never did), so **0.58 may return `unstable_timing`**; that
is a probe, not a candidate.

## Finished - do not spend slots here

Kernels: GEMMs at 85% of achievable streaming; the ~144 small kernels at their
1.1 us launch + 0.6 us latency floor (splitting a row needs a second launch and
buys nothing); both directions of the split knob losing 4.3% (`c103` 512->128,
`c107` fill-the-slots); epilogue fusion dead on the CTA constraint (each buys
~1% of launches by forcing SPLITS=1 on the projection carrying the bytes);
megakernel bounded near 0.8% by the measured 1.0-1.3 us node cost; 8-warp
GEMMs; persistent lm_head; attention layouts reaching down in tile width
(`c108`).

Also closed: c106 speculate-above-16 (no hidden workload has more than 16
sequences); c102 mailbox view; c104 refine budget share; c105 lead-in replay;
fused argmax by default (it bypasses the layout search).

## Next, in order

1. Read the floor runs. If lowering the floor pays, push to 0.55 on SSS - that
   is where the engine becomes the limit.
2. The draft side is the rebuild, because **every draft dead end was measured
   under exact acceptance and no longer applies**: block size at batch 1
   (`block_candidates` returns `sizes[:1]`, and `DRAFTS_BY_MATCH` /
   `EXPECTED_PASSES` stop at 16), the sibling/chain split, `SPEC_LOOKAHEAD`.
   Build each on the best MEASURED relaxed tree, never on an unmeasured one.
