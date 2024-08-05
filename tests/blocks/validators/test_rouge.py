# Local
from fms_dgt.blocks.validators.rouge import RougeDedupValidator


class TestRougeValidator:
    def test_matches(self):
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=0.0)

        all_data = [
            "I went to the store yesterday",
            "blue red yellow green",
        ]
        all_tokens = validator.tokenize(all_data)

        data_entry = "I went to the store"
        new_tokens1 = validator.tokenize(data_entry)
        data_entry = "I went to the store yesterday"
        new_tokens2 = validator.tokenize(data_entry)
        inputs = [{"a": new_tokens1}, {"a": new_tokens2}]
        validator._threshold = 0.91
        validator.generate(
            inputs, context=all_tokens, arg_fields=["a"], result_field="result"
        )
        assert inputs[0]["result"] and not inputs[1]["result"]

        data_entry = "one two three"
        new_tokens = validator.tokenize(data_entry)
        inputs = [{"a": new_tokens}]
        validator._threshold = 0.0
        validator.generate(
            inputs, context=all_tokens, arg_fields=["a"], result_field="result"
        )
        assert not inputs[0]["result"]
