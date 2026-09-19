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
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb

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
    from kernels.rmsnorm import add_rms_norm, rms_norm
    from kernels.swiglu import swiglu
    from kernels.qk_rope import qk_rope_cache
    from kernels.decode_attention import decode_attention
    from kernels.linear import _project

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

        for rows, width, tokens in ((1, 2560, 1), (16, 2560, 1), (31, 256, 1), (2, 2560, 7)):
            x = torch.randn(rows, tokens, width, device="cuda", dtype=torch.bfloat16)[:, -1:, :]
            residual = torch.randn_like(x)
            original_x, original_residual = x.clone(), residual.clone()
            norm = Qwen3RMSNorm(width, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
            norm.weight.copy_(torch.randn_like(norm.weight))
            expected_sum = x + residual
            expected = norm(expected_sum)
            actual, actual_sum = add_rms_norm(x, residual, norm.weight, norm.variance_epsilon)
            assert actual.shape == x.shape
            assert torch.equal(x, original_x) and torch.equal(residual, original_residual)
            assert torch.equal(actual_sum, expected_sum)
            assert (actual == expected).float().mean().item() >= 0.999
            torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)
        print("Fused residual-add + RMSNorm parity: passed", flush=True)

        for rows, width in ((1, 9728), (16, 9728), (19, 512)):
            packed = torch.randn(rows, 2 * width, dtype=torch.bfloat16, device="cuda")
            packed[:, :8] = torch.tensor(
                [-100, -20, -1, 0, 1, 20, 100, 0.125], device="cuda", dtype=packed.dtype
            )
            packed = packed.unsqueeze(0)  # Exercise [batch, tokens, 2*I].
            gate, up = packed.chunk(2, dim=-1)
            expected = torch.nn.functional.silu(gate) * up
            actual = swiglu(packed)
            assert (actual == expected).float().mean().item() >= 0.999
            torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.008)
        print("SwiGLU BF16 cast parity: passed", flush=True)

        for batch, q_heads, kv_heads, dim in ((1, 32, 8, 128), (4, 32, 8, 128), (2, 8, 2, 64)):
            packed = torch.randn(batch, 1, (q_heads + 2 * kv_heads) * dim, device="cuda", dtype=torch.bfloat16)
            q_norm = Qwen3RMSNorm(dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
            k_norm = Qwen3RMSNorm(dim, eps=1e-3).to(device="cuda", dtype=torch.bfloat16)
            q_norm.weight.copy_(torch.randn_like(q_norm.weight))
            k_norm.weight.copy_(torch.randn_like(k_norm.weight))
            q, k, v = packed.split((q_heads * dim, kv_heads * dim, kv_heads * dim), dim=-1)
            phase = torch.randn(1, 1, dim // 2, device="cuda")
            phase = torch.cat((phase, phase), dim=-1)
            cos, sin = phase.cos().bfloat16(), phase.sin().bfloat16()
            norm_q = q_norm(q.reshape(batch, 1, q_heads, dim)).transpose(1, 2)
            norm_k = k_norm(k.reshape(batch, 1, kv_heads, dim)).transpose(1, 2)
            expected_q, expected_k = apply_rotary_pos_emb(norm_q, norm_k, cos, sin)
            rotated_q = torch.cat((-norm_q[..., dim // 2:], norm_q[..., :dim // 2]), dim=-1)
            wrong_rope = (
                norm_q.float() * cos.unsqueeze(1).float()
                + rotated_q.float() * sin.unsqueeze(1).float()
            ).bfloat16()
            assert (wrong_rope != expected_q).float().mean().item() > 0.01
            keys = torch.full((batch, kv_heads, 9, dim), 17, device="cuda", dtype=torch.bfloat16)
            values = torch.full_like(keys, -13)
            position = torch.tensor([3], device="cuda", dtype=torch.int64)
            actual_q = qk_rope_cache(packed, q_norm, k_norm, cos, sin, position, keys, values, q_heads)
            for actual, expected in ((actual_q, expected_q), (keys[:, :, 3:4, :], expected_k)):
                assert (actual == expected).float().mean().item() >= 0.998
                torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.008)
            assert torch.equal(values[:, :, 3:4, :], v.reshape(batch, 1, kv_heads, dim).transpose(1, 2))
            for tensor, sentinel in ((keys, 17), (values, -13)):
                assert torch.all(tensor[:, :, :3, :] == sentinel)
                assert torch.all(tensor[:, :, 4:, :] == sentinel)
        print("Fused Q/K norm + RoPE + cache-write parity and untouched slots: passed", flush=True)

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

        for batch, kv_heads, dim, capacity, valid in (
            (1, 8, 128, 1, 1), (1, 8, 128, 65, 1), (1, 8, 128, 65, 33),
            (1, 8, 128, 544, 513), (4, 8, 128, 2080, 2048),
            (16, 8, 128, 640, 639), (2, 2, 64, 65, 65),
        ):
            for magnitude in (1.0, 3.0):
                q = torch.randn(batch, kv_heads * 4, 1, dim, device="cuda", dtype=torch.bfloat16) * magnitude
                k = torch.randn(batch, kv_heads, capacity, dim, device="cuda", dtype=torch.bfloat16) * magnitude
                v = torch.randn_like(k)
                mask = (torch.arange(capacity, device="cuda") < valid).view(1, 1, 1, -1)
                expected, _ = sdpa_attention_forward(module, q, k, v, mask, scaling=dim**-0.5)
                # Unused memory must never enter the dot product, even if it
                # holds NaN rather than the engine's finite stale cache values.
                k[:, :, valid:, :] = float("nan")
                v[:, :, valid:, :] = float("nan")
                position = torch.tensor([valid - 1], device="cuda", dtype=torch.int64)
                actual = decode_attention(q, k, v, position, dim**-0.5)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
        print("Dense split-KV decode attention parity, partial/empty splits and unused NaNs: passed", flush=True)

        for rows, outputs, width in ((1, 173, 259), (4, 257, 513), (16, 127, 256), (1, 6144, 2560), (4, 2560, 9728)):
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(outputs, width, device="cuda", dtype=torch.bfloat16) * 0.02
            expected = torch.nn.functional.linear(x, weight)
            configs = [("gemm", 64, 128, 1), ("gemm", 64, 128, 4)]
            if rows == 1:
                configs.append(("gemv", 8, 512, 1))
            for config in configs:
                actual = _project(x, weight, config)
                torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.016)
        print("BF16 projections: scalar/tensor-core, split-K, ragged N/K parity passed", flush=True)

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
                num_hidden_layers=2, num_attention_heads=8,
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
