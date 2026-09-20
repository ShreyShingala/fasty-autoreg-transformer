# Handoff — Dryft Kernel Rush, team SSS (for GPT Astra)

You are taking over an inference-engine optimisation loop that is currently
**#2 on the leaderboard**, 4.8% behind Segfault. Everything below is fact as of
2026-09-20 01:00 UTC. The three queues we still use are SSS, dryfter and
0xDeadBeaf; Silver Bullet has been handed to another agent building a new
architecture from scratch, so do not dispatch to it.

## The task

Make `engine/` (the only thing submitted) produce the model's **exact greedy
tokens** as fast as possible for Qwen3-4B-Instruct-2507 BF16 on one H100.
Score = geometric mean of tokens/s over six hidden workloads (fixed batch,
prompt, output; public probes: 1x512->32, 4x2048->32, 16x512->128). Read
`AGENTS.md` once in full — it is the contract. Hard rules: PyTorch 2.5.1 +
Triton 3.1.0 + Transformers 4.51.3 only (no vLLM/flash-attn/CUDA/C++), no
network, no extra weights, no quantisation or approximation, every token must
be native's argmax or within 2.0 logits of it (a teacher-forced replay checks
this after the run), TTFT/TPOT <= 1.10x native, <= 25% spread across the five
samples, <= 90% memory, and the WHOLE run (six workloads: load + warmup +
samples) is killed at **900 s** — we have been killed twice by this.

## Where we are — 2026-09-20

| | score | note |
|---|---|---|
| **Segfault** | **1198.9** | took the lead; went 1064.4 -> 1176.4 -> 1198.9 in about an hour |
| SSS (us) | 1144.3 | `c758faf` (the base tree); we are #2 by 4.8% |
| 0xDeadBeaf | 1142.9 | fourth queue, ours |
| dryfter | 1138.7 | merged team |
| Silver Bullet | 1137.7 | **handed to another agent for a from-scratch architecture** |

### Two measurement rules that matter more than anything else here

**1. Paired draws of the same tree agree to ~0.1-0.2% normalized.** Base
1143.7/1143.6, candidate 106 1139.8/1140.9, refine-share 1121.5/1122.1, fused
argmax 1123.6/1121.0. The often-quoted 1.33% sigma is the spread of the RAW
score and is dominated by node speed; `report_runs.py` divides it out using
native's own prefill TTFT measured in the same run. So **a 0.5% change is
readable in one run** — the old "needs 3%" bar is wrong. Discard any run whose
`node` column is more than ~5% off: one drew a 30%-slow node and scored 1053
on the byte-identical base tree.

**2. Public tokens/s moves OPPOSITE to the score. Do not use it.** The base has
nearly the lowest public-0 throughput of twelve measured trees (307.6
normalized) and the highest score. Every change we measured as a ~2% regression
is 4-6% FASTER on the batch-1 public probe. Only the hidden workloads are
scored, and they respond to something the public probes do not.

### The thing to understand before changing anything

Twelve trees measured. **Only two did not lose: `num_stages` 2->3 (neutral in
its own right) and candidate 106 (which only affects shapes above 16 sequences,
and the hidden set has none).** Everything substantive cost 1.9-4.3%, clustered
tightly near -2% — including two changes proven bit-identical offline, one of
which provably cuts memory instructions 8x.

Nine independent mechanisms in six files do not all cost the same 1.9% by
coincidence. The working hypothesis is that the hidden score is set by **which
layouts warmup tuning settles on**: any edit perturbs compile and timing order,
`_choose` and `refine` then judge slightly different things, and the base's
outcome happens to be the good one. If that is right, incremental edits are
taxed ~2% before they start, and only a change worth clearly more than that can
show up.

## How the engine works (read the code, this is the map)

`engine/engine.py` -> `Engine.__init__` (load, build the successor table,
`optimize_model`) and `generate` (yields one list of token ids per step,
exactly `max_new_tokens` times). `engine/decode.py` is the heart:
`DecodeState` allocates everything, captures a **prefill graph** and a
**verify-pass graph**, tunes layouts at warmup, then streams tokens.

**Exact self-speculative decoding** is what got us from 933 to 1130:
each CUDA-graphed *verify pass* processes B rows x T block tokens =
`[trusted token, D chain drafts, T-1-D sibling alternatives for draft 1]`.
Drafts come from (a) the row's own history via longest-suffix n-gram match,
(b) a model-derived top-8 successor table built at load, (c) the previous
pass's own prediction after its first wrong draft ("stale guess", lane 0).
`kernels/spec.py` proposes/settles/relocates on the GPU; the tree mask lives
in `kernels/decode_attention.py::_block_partials`. Acceptance is ~1.33-1.7
tokens/pass on platform text. **Release pacing** (`pace_floor`, PACE_FLOOR
0.70) holds tokens back so the five samples stay within the 25% spread gate —
at batch 1 the median sample sits on that floor, so *pass time* converts
almost 1:1 into score there.

Kernels: fused add+RMSNorm, QK-norm+RoPE+KV-write, SwiGLU, split-K skinny
GEMM with FP32 accumulation (`kernels/linear.py` + `kernels/gemm.py`, several
"kinds" judged at warmup and re-judged inside the captured graph by
`DecodeState.refine`), dense tree-mask attention, two-stage argmax, and
**Hopper TMA descriptor loads** for GEMM weight tiles (the one big recent win:
candidate 67 cut batch-4/16 pass time 4-7%).

## Local gates — run ALL of these before every push (~3 min, no GPU needed)

```
python3 -m unittest discover -s tests
set -a; . ./.env; set +a; ./bin/dryft validate engine      # never print .env
agent/local_cpu/interp/all.sh                              # every kernel EXECUTED on CPU
~/.cache/fasty-lab/venv/bin/python agent/local_cpu/smoke_engine.py   # whole engine, real 4-layer model vs HF greedy
docker run ... fasty-cpucheck:3.1.0 python /scratch/compile_<x>.py   # offline cuda:90 compile of any new kernel
```
`agent/local_cpu/README.md` has the docker commands. The Triton interpreter
image (`fasty-tritoninterp:3.5.0`) runs our kernels on CPU and is how we prove
a restructured kernel is **bit-identical** to the committed one. The MPS lab
(`~/.cache/fasty-lab`) ranks draft policies offline against the real model.

## What is in flight right now

- Measuring on SSS: `53e4a7a` (candidate 85: PDL off, fused lm_head+argmax
  knob, pinned-memory completion stamps, in-place RoPE tables).
- Queued on SSS: `425f95f` (candidates 86+87: plain-decode attention layout
  search for batches > 16; pass time for the pacing floor = fastest of five
  back-to-back groups).
- Dispatched: dryfter `33e665e` (candidate 86 tree), Silver Bullet `c3120ff`
  (candidate 85 tree, second draw).
- Held locally, gated: fused embedding+first-norm kernel (one launch per pass,
  interpreter-verified bit-identical).

## What is already dead (do NOT retry without a new mechanism)

Lab-killed on the real model: depth-2 draft trees, logit re-ranking of
siblings, hidden-state/PLD+ copy-source selection, pair/trigram successor
tables, copy-logit (RACER) siblings, token recycling within a generation,
layer-skip/early-exit/Jacobi self-drafting. **Even a perfect copy-source
selector saves only 3-5% of passes**: the draft side is at its ceiling for
training-free methods on this text.
Platform-killed: ranked-draft kernel, paired gate/up verify kernel, 32 MiB
cuBLAS workspace, cuBLASLt preference, prefill tuners (gated GEMM, cuDNN SDPA:
TTFT never moved in 25 runs), 64-row tile GEMM (blew the 900 s cap),
per-batch-class EXPECTED_PASSES refit (-1.5% hidden), **Programmatic Dependent
Launch** (+13% batch-1 TPOT: early blocks squat on SMs and starve the
bandwidth-bound kernel still running; the no-op wait instruction stays in the
kernels, `ENABLED = False`).

## Platform facts worth knowing

Host is gVisor: Triton compiles are slow (~55 specializations + 35-45 graph
captures per workload) and every CUDA event query is a trapped syscall.
Platform overhead is ~45 s per workload. **Engine stdout is hidden on purpose**
(hidden shapes could be encoded in text) — never build telemetry side channels;
that was proposed once and refused. Every warmup second costs six (one per
workload). Cancel a superseded run with `./bin/dryft cancel <id>`.

## Where to read more

`agent/continue.md` (state), `agent/EXPERIMENTS.md` (every candidate 8-88 with
its result and why it was kept or reverted — read the tail), `agent/NEXT_PLAN.md`
(ranked ideas and the dead list), `~/.cache/fasty-lab/plan/*.md` (about 20
research reports: pass cost model, warmup audit, pacing simulation, Triton 3.1
Hopper feature audit, web/Exa research), `agent/RESEARCH_PROMPT.md` (a
self-contained prompt for outside research).

## The loop

Pick one idea; write the hypothesis in `EXPERIMENTS.md`; kill it offline in the
lab if it is a draft idea; implement; run every local gate; push to SSS main
(that IS the experiment) or dispatch to a team repo; start the watcher
(`cd agent/tools && nohup python3 watch_run.py <sha> &`); immediately begin the
next idea. Collect with `python3 collect_runs.py`. Keep one run measuring and
one queued per queue. The user's instruction: **prefer many small verified
gains over one big architectural bet**, and never stop.
