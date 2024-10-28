# Standard
from typing import Dict, List
import copy
import os
import time

# Third Party
import pytest
import torch

# Local
from fms_dgt.base.registry import get_block
from fms_dgt.blocks.generators.llm import CachingLM, LMGenerator

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
    "tensor_parallel_size": 1,
    **GREEDY_CFG,
}
GREEDY_VLLM_SERVER_CFG = {
    "type": "vllm-server",
    "model_id_or_path": "ibm-granite/granite-8b-code-instruct",
    "tensor_parallel_size": 1,
    **GREEDY_CFG,
}
GREEDY_OPENAI_CFG = {
    "type": "openai-chat",
    "model_id_or_path": "gpt-3.5-turbo",
    **GREEDY_CFG,
}
GREEDY_WATSONX_CFG = {
    "type": "watsonx",
    "model_id_or_path": "granite-3-8b-instruct",
    **GREEDY_CFG,
}
PROMPTS = [f"Question: x = {i} + 1\nAnswer: x =" for i in range(25)]


@pytest.mark.parametrize(
    "model_cfg",
    [GREEDY_VLLM_CFG],  # GREEDY_OPENAI_CFG, GREEDY_GENAI_CFG]
)
def test_generate_batch(model_cfg):
    model_cfg = dict(model_cfg)
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm.generate(inputs, arg_fields=["prompt"], result_field="output")

    for i, inp in enumerate(inputs):
        assert (
            inp["prompt"] == inputs_copy[i]["prompt"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["output"], str)


@pytest.mark.parametrize("model_cfg", [GREEDY_GENAI_CFG])  # , GREEDY_VLLM_CFG])
def test_loglikelihood_batch(model_cfg):
    model_cfg = dict(model_cfg)
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt1": prompt, "prompt2": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm.generate(
        inputs,
        arg_fields=["prompt1", "prompt2"],
        result_field="output",
        method="loglikelihood",
    )

    for i, inp in enumerate(inputs):
        assert (
            inp["prompt1"] == inputs_copy[i]["prompt1"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["output"], float)


# def test_loglikelihood_batch_alignment():
#     vllm_config, genai_config = dict(GREEDY_VLLM_CFG), dict(GREEDY_GENAI_CFG)
#     vllm_config["model_id_or_path"] = "ibm-granite/granite-8b-code-instruct"
#     genai_config["model_id_or_path"] = "ibm/granite-8b-code-instruct"

#     vllm: LMGeneratorBlock = get_block(vllm_config["type"],
#         name=f"test_{vllm_config['type']}", config=vllm_config
#     )
#     genai: LMGeneratorBlock = get_block(genai_config["type"],
#         name=f"test_{genai_config['type']}", config=genai_config
#     )

#     inputs: List[Instance] = []
#     for prompt in PROMPTS[:1]:
#         args = [prompt, prompt]
#         inputs.append(Instance(args))

#     inputs_vllm = copy.deepcopy(inputs)
#     inputs_genai = copy.deepcopy(inputs)

#     vllm.loglikelihood_batch(inputs_vllm)
#     genai.loglikelihood_batch(inputs_genai)

#     for i, inp in enumerate(inputs):
#         assert (
#             inp.args == inputs_vllm[i].args == inputs_genai[i].args
#         ), f"Input list has been rearranged at index {i}"


def test_lm_caching():
    cache_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tmp_cache.db"
    )
    if os.path.exists(cache_path):
        os.remove(cache_path)

    model_cfg = dict(GREEDY_GENAI_CFG)
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    non_cache_inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt}
        non_cache_inputs.append(inp)

    pre_cache_inputs = copy.deepcopy(non_cache_inputs)
    post_cache_inputs = copy.deepcopy(non_cache_inputs)

    non_cache_time = time.time()
    lm.generate(non_cache_inputs, arg_fields=["prompt"], result_field="output")
    non_cache_time = time.time() - non_cache_time

    cache_lm = CachingLM(
        lm,
        cache_path,
    )

    pre_cache_time = time.time()
    cache_lm.generate(pre_cache_inputs, arg_fields=["prompt"], result_field="output")
    pre_cache_time = time.time() - pre_cache_time

    post_cache_time = time.time()
    cache_lm.generate(post_cache_inputs, arg_fields=["prompt"], result_field="output")
    post_cache_time = time.time() - post_cache_time

    os.remove(cache_path)

    assert (
        post_cache_time < pre_cache_time and post_cache_time < non_cache_time
    ), f"Caching led to increased execution time {(post_cache_time, pre_cache_time, non_cache_time)}"

    for i, (non, pre, post) in enumerate(
        zip(non_cache_inputs, pre_cache_inputs, post_cache_inputs)
    ):
        assert (
            non["prompt"] == pre["prompt"] == post["prompt"]
        ), f"Input list has been rearranged at index {i}: {(non['prompt'], pre['prompt'], post['prompt'])}"
        assert (
            non["output"] == pre["output"] == post["output"]
        ), f"Different results detected at index {i}: {(non['output'], pre['output'], post['output'])}"

    def test_vllm_free_model_memory(self):
        model_cfg = dict(GREEDY_VLLM_CFG)
        model_cfg["type"] = "vllm"
        model_cfg["tensor_parallel_size"] = 1
        model_cfg["model_id_or_path"] = "ibm-granite/granite-8b-code-instruct"
        model_type = model_cfg.get("type")

        # first we test generation
        inputs: List[Dict] = []
        for prompt in PROMPTS[:3]:
            inp = {"prompt": prompt}
            inputs.append(inp)

        lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)
        lm1_outputs = lm.generate(
            copy.deepcopy(inputs), arg_fields=["prompt"], result_field="output"
        )

        # check memory, release, and reinitialize
        before_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        lm.release_model()
        after_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        lm.init_model()

        assert (
            before_mem > after_mem * 10
        ), f"Expected more of a difference in memory usage after model memory released: {before_mem} GB vs {after_mem} GB"

        lm2_outputs = lm.generate(
            copy.deepcopy(inputs), arg_fields=["prompt"], result_field="output"
        )

        for i, inp in enumerate(lm1_outputs):
            assert (
                inp["prompt"] == lm2_outputs[i]["prompt"]
            ), f"Input list has been rearranged at index {i}"
            assert (
                isinstance(inp["output"], str)
                and inp["output"] == lm2_outputs[i]["output"]
            )


def test_vllm_remote_batch():
    """
    start server with

    python -m vllm.entrypoints.openai.api_server --model ibm-granite/granite-8b-code-instruct

    """
    model_cfg = dict(GREEDY_VLLM_CFG)
    model_cfg["type"] = "vllm-remote"
    model_cfg["base_url"] = "http://0.0.0.0:8000/v1"
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # first we test generation
    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm.generate(inputs, arg_fields=["prompt"], result_field="output")

    for i, inp in enumerate(inputs):
        assert (
            inp["prompt"] == inputs_copy[i]["prompt"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["output"], str)

    # now we test loglikelihood
    # inputs: List[Dict] = []
    # for prompt in PROMPTS:
    #     inp = {"prompt1": prompt, "prompt2": prompt}
    #     inputs.append(inp)

    # inputs_copy = copy.deepcopy(inputs)

    # lm.generate(
    #     inputs,
    #     arg_fields=["prompt1", "prompt2"],
    #     result_field="output",
    #     method="loglikelihood",
    # )

    # for i, inp in enumerate(inputs):
    #     assert (
    #         inp["prompt1"] == inputs_copy[i]["prompt1"]
    #     ), f"Input list has been rearranged at index {i}"
    #     assert isinstance(inp["output"], float)


def test_vllm_tensor_parallel():
    """

    replace "model_id_or_path" with suitably large model and ensure you have 2 GPUs of sufficient size, e.g. 2 of the a100_80gb

    """
    model_cfg = dict(GREEDY_VLLM_CFG)
    model_cfg["type"] = "vllm"
    model_cfg["tensor_parallel_size"] = 2
    model_cfg["model_id_or_path"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model_type = model_cfg.get("type")
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # first we test generation
    inputs: List[Dict] = []
    for prompt in PROMPTS:
        inp = {"prompt": prompt}
        inputs.append(inp)

    inputs_copy = copy.deepcopy(inputs)

    lm.generate(inputs, arg_fields=["prompt"], result_field="output")

    for i, inp in enumerate(inputs):
        assert (
            inp["prompt"] == inputs_copy[i]["prompt"]
        ), f"Input list has been rearranged at index {i}"
        assert isinstance(inp["output"], str)


@pytest.mark.parametrize("model_cfg", [GREEDY_GENAI_CFG, GREEDY_OPENAI_CFG])
def test_auto_chat_template(model_cfg):
    model_type = model_cfg.get("type")
    model_cfg["auto_chat_template"] = True
    model_cfg["model_id_or_path"] = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    lm: LMGenerator = get_block(model_type, name=f"test_{model_type}", **model_cfg)

    # check it passes through for a simple string
    prompt = {"prompt": "Hello world"}
    outputs, _ = lm.get_args_kwargs(prompt, lm.GENERATE, ["prompt"])
    output = outputs[0]
    if "openai" in model_type:
        assert output == prompt["prompt"]
    else:
        assert output != prompt["prompt"]

    # check it passes through a list of dictionaries
    prompt = {
        "prompt": [
            {"role": "user", "content": "Hello World"},
            {"role": "assistant", "content": "Yes, it is me, World"},
        ]
    }
    outputs, _ = lm.get_args_kwargs(prompt, lm.GENERATE, ["prompt"])
    output = outputs[0]
    if "openai" in model_type:
        assert output == prompt["prompt"]
    else:
        assert output != prompt["prompt"]

    # check it does nothing for loglikelihood
    prompt = {"prompt": "Hello world"}
    outputs, _ = lm.get_args_kwargs(prompt, lm.LOGLIKELIHOOD, ["prompt"])
    output = outputs[0]
    assert output == prompt["prompt"]
