# Standard
from typing import List
import copy
import os
import time

# Third Party
import pytest

# Local
from fms_dgt.base.instance import Instance
from fms_dgt.blocks.validators.lm_judge import LMJudgeValidator

GREEDY_CFG = {
    "lm_config": {
        "type": "genai",
        "decoding_method": "greedy",
        "temperature": 0.7,  # oddly, I get less variation with 1.0
        "max_new_tokens": 100,
        "min_new_tokens": 1,
        "model_id_or_path": "mistralai/mixtral-8x7b-instruct-v01",
    }
}


def showit(x):
    # print("HERE IS THE OUTPUT:",x, "DONE OUTPUT")
    return True


class TestLlmJudgeValidator:
    @pytest.mark.parametrize("model_backend", ["genai"])
    def test_generate_batch(self, model_backend):
        lm_judge = LMJudgeValidator(name=f"test_{model_backend}", **GREEDY_CFG)

        inp = "Pick a random word from the following list: Apple, Orange, Pineapple, Cranberry, Watermelon and Mango"
        ninp = 6  # Oddly, only the  first tends to be different
        inputs = [{"lm_input": inp, "success_func": showit} for i in range(ninp)]

        lm_judge.generate(
            inputs,
            arg_fields=["success_func"],
            lm_arg_fields=["lm_input"],
            result_field="result",
            lm_result_field="result",
        )
        for i in range(1, ninp):
            print(i)
            # if you print out the value returned from genai,
            # you'll see that it is not the same for all calls,
            # but what is returned here is only the first output.
            assert inputs[0]["result"] is inputs[i]["result"]
