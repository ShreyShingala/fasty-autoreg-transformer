"""CPU-only tests of the streamed Engine protocol, using a fake GPU backend.

These do not establish numerical correctness; agent/verify_gpu.py does that.
"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return Tensor([row[0] for row in self.values])

    def tolist(self):
        return list(self.values)


class InferenceMode:
    active = 0

    def __enter__(self):
        InferenceMode.active += 1

    def __exit__(self, *args):
        InferenceMode.active -= 1


class State:
    builds = 0

    def __init__(self, model, shape):
        State.builds += 1
        self.shape = shape
        self.device = "fake"
        self.graph = self

    def prefill(self, prompt):
        self.token_ids = Tensor([[row[-1]] for row in prompt.values])

    def replay(self):
        self.token_ids = Tensor([[row[0] + 1] for row in self.token_ids.values])


def load_engine():
    torch = ModuleType("torch")
    torch.int64 = "int64"
    torch.tensor = lambda values, **kwargs: Tensor(values)
    torch.inference_mode = InferenceMode
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = object
    decode = ModuleType("decode")
    decode.DecodeState = State
    decode.optimize_model = lambda model: None
    spec = importlib.util.spec_from_file_location("candidate_engine", ROOT / "engine/engine.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "decode": decode}):
        spec.loader.exec_module(module)
    return module.Engine


class EngineContractTests(unittest.TestCase):
    def setUp(self):
        cls = load_engine()
        self.engine = cls.__new__(cls)
        self.engine.model = object()
        self.engine.state = None
        State.builds = 0

    def test_yield_count_batch_order_and_eos(self):
        # Zero is a fake EOS. It must be streamed normally and decoding resumes.
        result = list(self.engine.generate([[8, 0], [9, 7]], 4))
        self.assertEqual(result, [[0, 7], [1, 8], [2, 9], [3, 10]])

    def test_single_output_never_replays_decode(self):
        self.assertEqual(list(self.engine.generate([[42]], 1)), [[42]])

    def test_zero_outputs_do_not_allocate(self):
        self.assertEqual(list(self.engine.generate([], 0)), [])
        self.assertEqual(State.builds, 0)

    def test_yield_does_not_leak_inference_context(self):
        generator = self.engine.generate([[2]], 3)
        self.assertEqual(next(generator), [2])
        self.assertEqual(InferenceMode.active, 0)
        generator.close()
        self.assertEqual(list(self.engine.generate([[8]], 2)), [[8], [9]])

    def test_repeated_shape_resets_prompt_and_reuses_state(self):
        list(self.engine.generate([[1, 2]], 3))
        self.assertEqual(list(self.engine.generate([[7, 8]], 3)), [[8], [9], [10]])
        self.assertEqual(State.builds, 1)

    def test_changed_shape_replaces_state(self):
        list(self.engine.generate([[2]], 2))
        self.assertEqual(list(self.engine.generate([[3], [4]], 1)), [[3, 4]])
        self.assertEqual(State.builds, 2)

    def test_invalid_prompt_shapes_rejected(self):
        for prompts in ([], [[]], [[1], [2, 3]]):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                list(self.engine.generate(prompts, 2))


if __name__ == "__main__":
    unittest.main()
