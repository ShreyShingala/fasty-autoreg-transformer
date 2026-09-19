"""Check the actual kernels, graph replay, and own-prefix logits on CUDA.

Run with the pinned runtime: python agent/verify_gpu.py [--model-path /weights]
Without a path, use a small randomly initialized Qwen3 (no downloads). With a
path, test the provided checkpoint on all three public shapes. This script is
not submitted and does not replace the platform's teacher-forced judge.
"""

import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()

    import torch
    import transformers
    import triton
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU protocol tests cannot validate GPU numerics")
    if (torch.__version__.split("+")[0], transformers.__version__, triton.__version__) != ("2.5.1", "4.51.3", "3.1.0"):
        raise SystemExit("Use PyTorch 2.5.1, Transformers 4.51.3 and Triton 3.1.0, matching the judge")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
    from engine import Engine
    from decode import optimize_model
    from attention import grouped_sdpa
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from kernels.rmsnorm import rms_norm

    with torch.inference_mode():
        # Exercise real Qwen widths, BF16 cast placement and non-contiguous input.
        for width in (128, 2560):
            for x in (
                torch.randn(9, width, device="cuda", dtype=torch.bfloat16),
                torch.randn(2, 3, width, device="cuda", dtype=torch.bfloat16)[:, -1:, :],
            ):
                weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
                native_norm = Qwen3RMSNorm(width, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
                native_norm.weight.copy_(weight)
                expected = native_norm(x)
                fp = x.float()
                wrong_cast = (
                    fp * torch.rsqrt(fp.square().mean(-1, keepdim=True) + 1e-6) * weight.float()
                ).to(x.dtype)
                assert (wrong_cast != expected).float().mean().item() > 0.05
                actual = rms_norm(x, weight, 1e-6)
                assert (actual == expected).float().mean().item() >= 0.999
                # A normalization rounding boundary followed by the weight
                # multiply can change the rounded product by up to two ULPs.
                ulps = (actual.contiguous().view(torch.int16).int() - expected.contiguous().view(torch.int16).int()).abs()
                assert ulps.max().item() <= 2
        print("RMSNorm BF16 parity: passed", flush=True)

        module = SimpleNamespace(num_key_value_groups=4)
        for batch, capacity, valid in ((1, 33, 1), (2, 65, 23), (4, 32, 32)):
            q = torch.randn(batch, 32, 1, 128, dtype=torch.bfloat16, device="cuda")
            k = torch.randn(batch, 8, capacity, 128, dtype=torch.bfloat16, device="cuda")
            v = torch.randn_like(k)
            mask = (torch.arange(capacity, device="cuda") < valid).view(1, 1, 1, -1)
            expected, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=128**-0.5)
            actual, _ = grouped_sdpa(module, q, k, v, mask, scaling=128**-0.5)
            torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
        print("Grouped SDPA parity against repeated KV heads: passed", flush=True)

        if args.model_path:
            reference = AutoModelForCausalLM.from_pretrained(
                str(args.model_path), torch_dtype=torch.bfloat16,
                attn_implementation="sdpa", local_files_only=True,
            ).eval().cuda()
            candidate = Engine(str(args.model_path))
            shapes = [(1, 512, 32), (4, 2048, 32), (16, 512, 128), (1, 512, 1)]
        else:
            config = Qwen3Config(
                vocab_size=512, hidden_size=256, intermediate_size=512,
                num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=64, max_position_embeddings=4096,
                rope_theta=5_000_000.0, tie_word_embeddings=True,
            )
            config._attn_implementation = "sdpa"
            reference = Qwen3ForCausalLM(config).eval().to(device="cuda", dtype=torch.bfloat16)
            candidate = Engine.__new__(Engine)
            candidate.model = copy.deepcopy(reference)
            optimize_model(candidate.model)
            candidate.state = None
            shapes = [(1, 1, 1), (2, 13, 7), (1, 19, 9), (2, 13, 7), (4, 8, 3)]

        for batch, length, outputs in shapes:
            shape = (batch, length, outputs)
            for sample in range(3):
                prompt = torch.randint(3, reference.config.vocab_size, (batch, length), device="cuda")
                # Poison stale slots with large FINITE values. Prompt slots must
                # be overwritten and future positions excluded on every replay.
                if sample and candidate.state is not None:
                    for cache in candidate.state.cache.keys + candidate.state.cache.values:
                        cache.fill_(100)
                emitted = list(candidate.generate(prompt.tolist(), outputs))
                assert len(emitted) == outputs and all(len(row) == batch for row in emitted)
                chosen = torch.tensor(emitted, device="cuda", dtype=torch.int64).T
                # Own-prefix teacher forcing, as the judge does. Row-wise replay
                # bounds logits memory even on the public batch-16 workload.
                worst_gap = 0.0
                for row in range(batch):
                    ids = torch.cat((prompt[row:row + 1], chosen[row:row + 1, :-1]), dim=1)
                    logits = reference(ids, use_cache=False, logits_to_keep=outputs).logits[0].float()
                    selected = logits.gather(1, chosen[row, :, None]).squeeze(1)
                    worst_gap = max(worst_gap, (logits.max(dim=-1).values - selected).max().item())
                # A 2-logit threshold is much too loose for random tiny models.
                limit = 2.0 if args.model_path else 0.02
                assert worst_gap <= limit, (shape, sample, worst_gap)
                print(f"shape={shape} sample={sample} max_logit_gap={worst_gap:.6f}: passed", flush=True)
        torch.cuda.synchronize()
        print("CUDA graph replay, changed shapes, stale cache masking and own-prefix parity: passed")


if __name__ == "__main__":
    main()
