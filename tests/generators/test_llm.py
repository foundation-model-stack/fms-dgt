# Standard
from typing import List
import copy
import os
import time

# Third Party
import pytest

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import get_generator
from fms_sdg.generators.llm import CachingLM, LMGenerator

GREEDY_CFG = {
    "decoding_method": "greedy",
    "temperature": 1.0,
    "max_new_tokens": 5,
    "min_new_tokens": 1,
}
GREEDY_GENAI_CFG = {
    "type": "genai",
    "model_id_or_path": "ibm/granite-8b-code-instruct",
    **GREEDY_CFG,
}
GREEDY_VLLM_CFG = {
    "type": "vllm",
    "model_id_or_path": "ibm/granite-8b-code-instruct",
    **GREEDY_CFG,
}
GREEDY_OPENAI_CFG = {
    "type": "openai",
    "model_id_or_path": "gpt-3.5-turbo",
    **GREEDY_CFG,
}
PROMPTS = [f"Question: x = {i} + 1\nAnswer: x =" for i in range(25)]


class TestLlmGenerators:
    # @pytest.mark.parametrize("model_backend", ["genai", "openai-chat"])
    # @pytest.mark.parametrize("model_cfg", [GREEDY_GENAI_CFG, GREEDY_OPENAI_CFG])
    @pytest.mark.parametrize("model_backend", ["openai-chat"])
    @pytest.mark.parametrize("model_cfg", [GREEDY_OPENAI_CFG])
    def test_generate_batch(self, model_backend, model_cfg):
        lm: LMGenerator = get_generator(model_backend)(
            name=f"test_{model_backend}", config=model_cfg
        )

        inputs: List[Instance] = []
        for prompt in PROMPTS:
            args = [prompt]
            inputs.append(Instance(args))

        inputs_copy = copy.deepcopy(inputs)

        lm.generate_batch(inputs)

        for i, inp in enumerate(inputs):
            assert (
                inp.args == inputs_copy[i].args
            ), f"Input list has been rearranged at index {i}"
            assert isinstance(inp.result, str)

    def test_lm_caching(self):
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tmp_cache.db"
        )
        if os.path.exists(cache_path):
            os.remove(cache_path)

        model_backend = "genai"
        lm: LMGenerator = get_generator(model_backend)(
            name=f"test_{model_backend}", config=GREEDY_CFG
        )

        non_cache_inputs: List[Instance] = []
        for prompt in PROMPTS:
            args = [prompt]
            non_cache_inputs.append(Instance(args))

        pre_cache_inputs = copy.deepcopy(non_cache_inputs)
        post_cache_inputs = copy.deepcopy(non_cache_inputs)

        non_cache_time = time.time()
        lm.generate_batch(non_cache_inputs)
        non_cache_time = time.time() - non_cache_time

        cache_lm = CachingLM(
            lm,
            cache_path,
        )

        pre_cache_time = time.time()
        cache_lm.generate_batch(pre_cache_inputs)
        pre_cache_time = time.time() - pre_cache_time

        post_cache_time = time.time()
        cache_lm.generate_batch(post_cache_inputs)
        post_cache_time = time.time() - post_cache_time

        os.remove(cache_path)

        assert (
            post_cache_time < pre_cache_time and post_cache_time < non_cache_time
        ), f"Caching led to increased execution time {(post_cache_time, pre_cache_time, non_cache_time)}"

        for i, (non, pre, post) in enumerate(
            zip(non_cache_inputs, pre_cache_inputs, post_cache_inputs)
        ):
            assert (
                non.args == pre.args == post.args
            ), f"Input list has been rearranged at index {i}: {(non.args, pre.args, post.args)}"
            assert (
                non.result == pre.result == post.result
            ), f"Different results detected at index {i}: {(non.result, pre.result, post.result)}"
