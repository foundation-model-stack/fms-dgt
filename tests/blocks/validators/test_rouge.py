# Standard
from typing import List
import json

# Third Party
import pytest

# Local
from fms_dgt.blocks.validators.rouge import RougeValidator


class TestRougeValidator:
    def test_matches(self):
        validator = RougeValidator(name="test_rouge_validator", threshold=0.0)

        all_data = [
            "I went to the store",
            "I went to the store yesterday",
            "blue red yellow green",
        ]
        all_tokens = validator.tokenize(all_data)

        data_entry = "I went to the store"
        new_tokens = validator.tokenize(data_entry)
        inputs = [{"a": new_tokens, "b": all_tokens}]
        validator._threshold = 0.91
        # since "I went to the store" is in the input,
        # its similarity score with itself will be 1.0,
        # so the return value is False
        # (1.0 < 0.91)
        validator.generate(inputs, arg_fields=["a", "b"], result_field="result")
        assert not inputs[0]["result"]

        data_entry = "one two three"
        new_tokens = validator.tokenize(data_entry)
        inputs = [{"a": new_tokens, "b": all_tokens}]

        # a threshold of 0.0 ensures that the return value is False
        # (x<0.0) where x is non-negative
        # (since the similarity of data_entry with the items in add_data is 0.0,
        # any positive value would result in a return value of True)
        validator._threshold = 0.0
        validator.generate(inputs, arg_fields=["a", "b"], result_field="result")
        assert not inputs[0]["result"]
