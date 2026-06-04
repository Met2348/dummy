"""Triton Python backend skeleton for an LLM."""
# Reference shape only — actual deployment swaps `mock_generate` for vllm.
import numpy as np


class TritonPythonModel:
    def initialize(self, args):
        # load model here
        self.model_name = args.get("model_name", "mock")

    def execute(self, requests):
        responses = []
        for req in requests:
            in_ids = req.inputs["input_ids"].as_numpy()
            n = req.inputs["max_new_tokens"].as_numpy()[0]
            out = self._mock_generate(in_ids, n)
            responses.append({"output_ids": out})
        return responses

    def _mock_generate(self, ids, n):
        # Cycle through canned tokens
        return np.tile(ids[-1:], n).astype(np.int32)
