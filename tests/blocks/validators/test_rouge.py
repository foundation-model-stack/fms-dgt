# Local
from fms_dgt.blocks.validators.rouge import RougeDedupValidator


class TestRougeValidator:
    def test_matches(self):

        all_data = [
            "I went to the store yesterday",
            "blue red yellow green",
        ]

        inputs = [
            {"a": "I went to the store"},
            {"a": "I went to the store yesterday"},
        ]
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=0.91)
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        assert inputs[0]["result"] and not inputs[1]["result"]

        inputs = [
            {"a": "I went to the store"},
            {"a": "I went to the store"},
            {"a": "I went to the store yesterday"},
        ]
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=1.0)
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        assert (
            inputs[0]["result"] and not inputs[1]["result"] and not inputs[2]["result"]
        )

        validator = RougeDedupValidator(name="test_rouge_validator", threshold=None)
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        assert inputs[0]["result"] and inputs[1]["result"] and inputs[2]["result"]

        inputs = [{"a": "one two three"}]
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=0.0)
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        assert not inputs[0]["result"]
