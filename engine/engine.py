"""BF16 Qwen3 with a reusable KV cache and one CUDA graph per decode shape."""

import torch
from transformers import AutoModelForCausalLM

from decode import DecodeState, optimize_model


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        optimize_model(self.model)
        self.state = None

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if max_new_tokens <= 0:
            return
        if not input_ids or not input_ids[0]:
            raise ValueError("generate requires a nonempty batch and prompt")
        shape = (len(input_ids), len(input_ids[0]), max_new_tokens)
        if any(len(row) != shape[1] for row in input_ids):
            raise ValueError("all prompts must have the same length")

        with torch.inference_mode():
            if self.state is None or self.state.shape != shape:
                # The harness reuses a shape after warmup. Bound memory if a
                # local caller changes it rather than retaining many graphs.
                self.state = None
                self.state = DecodeState(self.model, shape)
            state = self.state
            prompt = torch.tensor(input_ids, dtype=torch.int64, device=state.device)
            state.prefill(prompt)
            tokens = state.token_ids[:, 0].tolist()
        yield tokens
        for _ in range(max_new_tokens - 1):
            with torch.inference_mode():
                state.graph.replay()
                tokens = state.token_ids[:, 0].tolist()
            yield tokens
