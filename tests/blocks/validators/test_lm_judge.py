# Standard
from typing import List
import copy
import os
import time

# Third Party
import pytest

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.validators.lm_judge import LMJudgeValidator

GREEDY_CFG = {
    "lm_type": "genai",
    "decoding_method": "greedy",
    "temperature": 1.0,
    "max_new_tokens": 25,
    "min_new_tokens": 1,
    "model_id_or_path": "ibm/granite-8b-code-instruct",
}


class TestLlmJudgeValidator:
    @pytest.mark.parametrize("model_backend", ["genai"])
    def test_generate_batch(self, model_backend):
        lm_judge = LMJudgeValidator(name=f"test_{model_backend}", config=GREEDY_CFG)

        inputs = [
            Instance(
                [
                    "Question: 1 + 1 = ?\nAnswer: ",
                    lambda x: any([num in x for num in ["2"]]),
                ]
            )
        ]
        lm_judge.validate_batch(inputs)
        assert inputs[0].result, "Result should be true!"

        inputs = [
            Instance(
                [
                    "Repeat the following exactly: 'this is true'\n",
                    lambda x: "false" in x,
                ]
            )
        ]
        lm_judge.validate_batch(inputs)
        assert not inputs[0].result, "Result should be false!"
