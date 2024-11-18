# Local
from fms_dgt.blocks.validators.rouge import RougeDedupValidator


class TestRougeValidator:
    def test_matches(self):

        all_data = [
            "I went to the store yesterday",
            "blue red yellow green",
        ]

        inputs = [
            {"input": "I went to the store"},
            {"input": "I went to the store yesterday"},
        ]
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=0.91)
        validator(inputs, context=all_data)
        assert inputs[0]["is_valid"] and not inputs[1]["is_valid"]

        inputs = [
            {"input": "I went to the store"},
            {"input": "I went to the store"},
            {"input": "I went to the store yesterday"},
        ]
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=1.0)
        validator(inputs, context=all_data)
        assert (
            inputs[0]["is_valid"]
            and not inputs[1]["is_valid"]
            and not inputs[2]["is_valid"]
        )

        validator = RougeDedupValidator(name="test_rouge_validator", threshold=None)
        validator(inputs, context=all_data)
        assert inputs[0]["is_valid"] and inputs[1]["is_valid"] and inputs[2]["is_valid"]

        inputs = [{"input": "one two three"}]
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=0.0)
        validator(inputs, context=all_data)
        assert not inputs[0]["is_valid"]
