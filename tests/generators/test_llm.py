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

# hf cache

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
os.environ["HF_HOME"] = os.path.join(BASE_PATH, ".cache", "huggingface", "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(
    BASE_PATH, ".cache", "huggingface", "datasets"
)

#

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
    "model_id_or_path": "ibm-granite/granite-8b-code-instruct",
    **GREEDY_CFG,
}
GREEDY_OPENAI_CFG = {
    "type": "openai-chat",
    "model_id_or_path": "gpt-3.5-turbo",
    **GREEDY_CFG,
}
PROMPTS = [f"Question: x = {i} + 1\nAnswer: x =" for i in range(25)]


class TestLlmGenerators:
    @pytest.mark.parametrize(
        "model_cfg", [GREEDY_GENAI_CFG, GREEDY_OPENAI_CFG]
    )  # GREEDY_VLLM_CFG]
    def test_generate_batch(self, model_cfg):
        lm: LMGenerator = get_generator(model_cfg["type"])(
            name=f"test_{model_cfg['type']}", config=model_cfg
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

    @pytest.mark.parametrize("model_cfg", [GREEDY_GENAI_CFG])  # , GREEDY_VLLM_CFG])
    def test_loglikelihood_batch(self, model_cfg):
        lm: LMGenerator = get_generator(model_cfg["type"])(
            name=f"test_{model_cfg['type']}", config=model_cfg
        )

        inputs: List[Instance] = []
        for prompt in PROMPTS:
            args = [prompt, prompt]
            inputs.append(Instance(args))

        inputs_copy = copy.deepcopy(inputs)

        lm.loglikelihood_batch(inputs)

        for i, inp in enumerate(inputs):
            assert (
                inp.args == inputs_copy[i].args
            ), f"Input list has been rearranged at index {i}"
            assert isinstance(inp.result, float)

    def test_loglikelihood_batch_alignment(self):
        vllm_config, genai_config = dict(GREEDY_VLLM_CFG), dict(GREEDY_GENAI_CFG)
        vllm_config["model_id_or_path"] = "ibm-granite/granite-8b-code-instruct"
        genai_config["model_id_or_path"] = "ibm/granite-8b-code-instruct"

        vllm: LMGenerator = get_generator(vllm_config["type"])(
            name=f"test_{vllm_config['type']}", config=vllm_config
        )
        genai: LMGenerator = get_generator(genai_config["type"])(
            name=f"test_{genai_config['type']}", config=genai_config
        )

        inputs: List[Instance] = []
        for prompt in PROMPTS[:1]:
            args = [prompt, prompt]
            inputs.append(Instance(args))

        inputs_vllm = copy.deepcopy(inputs)
        inputs_genai = copy.deepcopy(inputs)

        vllm.loglikelihood_batch(inputs_vllm)
        genai.loglikelihood_batch(inputs_genai)

        for i, inp in enumerate(inputs):
            assert (
                inp.args == inputs_vllm[i].args == inputs_genai[i].args
            ), f"Input list has been rearranged at index {i}"

    def test_lm_caching(self):
        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tmp_cache.db"
        )
        if os.path.exists(cache_path):
            os.remove(cache_path)

        lm: LMGenerator = get_generator(GREEDY_GENAI_CFG["type"])(
            name=f"test_{GREEDY_GENAI_CFG['type']}", config=GREEDY_GENAI_CFG
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
