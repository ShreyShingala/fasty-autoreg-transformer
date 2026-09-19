---
name: kernel-rush-loop
description: Run the Dryft Kernel Rush optimization loop for the Qwen3 engine in fasty-autoreg-transformer - build a candidate, check it without a GPU, push (a push is an official H100 run), watch runs in the background, rank draft policies in the local model lab, and record results. Use when asked to improve the engine's score, "keep cooking", queue candidates, or check the leaderboard.
---

# Kernel Rush loop

Repo: `~/Downloads/Coding/fasty-autoreg-transformer`. Read `agent/continue.md` first (state, best
commit, risks), then the tail of `agent/EXPERIMENTS.md`. Only `engine/` is submitted.

## The loop (never idle)

1. **One hypothesis per candidate.** Write it in `agent/EXPERIMENTS.md` before pushing: expected
   effect, which public number will show it, fallback commit.
2. **No-GPU checks** (this Mac has no CUDA), all must pass before a push:
   - `python3 -m unittest discover -s tests`
   - `set -a; . ./.env; set +a; ./bin/dryft validate engine` (never print `.env`)
   - Triton kernels: offline compile for H100 in Docker image `fasty-cpucheck:3.1.0`:
     `docker run --rm --platform linux/amd64 -v "$PWD/engine":/work/engine:ro -v "$PWD/agent/local_cpu":/scratch:ro -e PYTHONPATH=/work/engine:/scratch fasty-cpucheck:3.1.0 python /scratch/<script>.py`
     (`offline_compile.py` helper; `TRITON_INTERPRET` does NOT work there).
   - New index/mask/bookkeeping logic: emulate the kernel formulas line by line in plain Python
     against a reference (`agent/local_cpu/check_*.py`, `test_speculate.py`, `test_spec_queue.py`).
3. **Push = official run** (~8-10 min, queued FIFO, 15-minute hard cap per run, stdout hidden).
   `git add <explicit paths>; git commit; git push origin HEAD`. Never stage
   `agent/ATTACK_PLAN.md` or `agent/CLAUDE_SYSTEM_PROMPT.md`. Failed runs do not replace the best.
4. **Watch in the background, keep building**:
   `cd agent/tools && nohup python3 watch_run.py <short-sha> > /tmp/watch_<sha>.log 2>&1 &`
   It saves `agent/results/<run>.json` and prints score, per-public-case tokens/s, TTFT, TPOT,
   p10/p50/p90. Leaderboard: `python3 -c "from dryft_api import get; print(get('/api/v1/challenges/decode/leaderboard'))"`.
   Do not wait for a result before starting the next candidate; stack independent candidates.
5. **Read results as evidence, not proof.** Identical code varies ~±1% in score. public-0 is batch 1,
   public-1 batch 4 x 2048 prompt, public-2 batch 16 x 128 outputs; the hidden six decide the score.
   Record every result (wins, losses, failures) in `agent/EXPERIMENTS.md`, then update `agent/continue.md`.

## Draft-policy lab (speculative decoding)

Acceptance depends only on the model's greedy text, so rank drafting ideas offline before spending a
run: `~/.cache/fasty-lab` (venv, pinned Qwen3-4B on MPS, `gen.py`/`gen2.py` to make greedy samples,
`successor_top8.pt`, simulators `evalmore.py`, `lab.py`, `REPORT.md`). Only one model process at a time.
Metric: verify passes per output token (lower is better); batch = max over rows.

## Reviews and research (parallel, bounded, read-only)

- Adversarial code review of each kernel change: a Claude subagent AND
  `codex exec -s read-only "<prompt>" < /dev/null > out.txt 2>&1` (without `< /dev/null` it hangs).
  Give reviewers the invariants and ask for file:line defects with a failing scenario.
- Research scouts (web) and lab analysts as background subagents with written deliverables.

## Hard rules

Exact greedy output only (no quantization/approximation); Triton-only GPU code; no cross-generation
token state; never reveal `.env`; never reset the dirty tree; do not spend money (e.g. Modal) without
asking. Known facts and dead ends are in `agent/EXPERIMENTS.md` — check there before retrying an idea.
