# Third Party
import pytest

# Local
from fms_dgt.blocks.validators.lm_judge import LMJudgeValidator

LM_RESULT_FIELD = "lm_generation"
JUDGE_RESULT_FIELD = "judge_result"
GREEDY_CFG = {
    "lm_config": {
        "type": "genai",
        "decoding_method": "greedy",
        "temperature": 1.0,
        "max_new_tokens": 25,
        "min_new_tokens": 1,
        "model_id_or_path": "ibm/granite-8b-code-instruct",
        "result_field": LM_RESULT_FIELD,
    }
}


class TestLlmJudgeValidator:
    @pytest.mark.parametrize("model_backend", ["genai"])
    def test_generate_batch(self, model_backend):
        lm_judge = LMJudgeValidator(
            name=f"test_{model_backend}", result_field=JUDGE_RESULT_FIELD, **GREEDY_CFG
        )

        inputs = [
            {
                "lm_input": "Question: 1 + 1 = ?\nAnswer: ",
                "success_func": lambda x: any([num in x for num in ["2"]]),
            }
        ]
        lm_judge(
            inputs,
            arg_fields=["success_func"],
            lm_arg_fields=["lm_input"],
            result_field=JUDGE_RESULT_FIELD,
            lm_result_field=LM_RESULT_FIELD,
        )
        assert inputs[0][JUDGE_RESULT_FIELD], "Result should be true!"

        inputs = [
            {
                "lm_input": "Repeat the following exactly: 'this is true'\n",
                "success_func": lambda x: "false" in x,
            }
        ]
        lm_judge(
            inputs,
            arg_fields=["success_func"],
            lm_arg_fields=["lm_input"],
            result_field=JUDGE_RESULT_FIELD,
            lm_result_field=LM_RESULT_FIELD,
        )
        assert not inputs[0][JUDGE_RESULT_FIELD], "Result should be false!"

        inputs = [
            {
                "lm_input": "Is 'eat' a verb?\nRespond with 'yes' or 'no.\n",
                "success_func": lambda x: "yes" in x,
            }
        ]
        lm_judge(
            inputs,
            arg_fields=["success_func"],
            lm_arg_fields=["lm_input"],
        )
        assert inputs[0][JUDGE_RESULT_FIELD], "Result should be true!"
        assert (
            "yes" in inputs[0][LM_RESULT_FIELD]
        ), "Result should contain the word 'yes'!"
