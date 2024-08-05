# Local
from fms_dgt.blocks.validators.rouge import RougeDedupValidator


class TestRougeValidator:
    def test_matches(self):
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=0.0)

        all_data = [
            "I went to the store yesterday",
            "blue red yellow green",
        ]

        inputs = [{"a": "I went to the store"}, {"a": "I went to the store yesterday"}]
        validator._threshold = 0.91
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        assert inputs[0]["result"] and not inputs[1]["result"]

        inputs = [{"a": "one two three"}]
        validator._threshold = 0.0
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        assert not inputs[0]["result"]
