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
    "lm": {
        "type": "genai",
        "decoding_method": "greedy",
        "temperature": 1.0,
        "max_new_tokens": 25,
        "min_new_tokens": 1,
        "model_id_or_path": "ibm/granite-8b-code-instruct",
    }
}


class TestLlmJudgeValidator:
    @pytest.mark.parametrize("model_backend", ["genai"])
    def test_generate_batch(self, model_backend):
        lm_judge = LMJudgeValidator(name=f"test_{model_backend}", **GREEDY_CFG)

        inputs = [
            {
                "lm_input": "Question: 1 + 1 = ?\nAnswer: ",
            }
        ]
        lm_judge.generate(
            inputs,
            arg_fields=["lm_input"],
            result_field="result",
            success_func=lambda x: any([num in x for num in ["2"]]),
        )
        assert inputs[0]["result"], "Result should be true!"

        inputs = [
            {
                "lm_input": "Repeat the following exactly: 'this is true'\n",
            }
        ]
        lm_judge.generate(
            inputs,
            arg_fields=["lm_input"],
            result_field="result",
            success_func=lambda x: "false" in x,
        )
        assert not inputs[0]["result"], "Result should be false!"
