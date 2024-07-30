# Standard
from typing import List
import json

# Third Party
import pytest

# Local
from fms_dgt.blocks.validators.dupchecker import DupCheckerValidator


class TestDupCheckerValidator:
    def test_matches(self):
        validator = DupCheckerValidator(name="test_dupchecker_validator")
        all_data = [
            "I went to the store",
            "I went to the store yesterday",
            "blue red yellow green",
        ]
        all_tokens = validator.tokenize(all_data)

        data_entry = "I went to the store"
        new_tokens = validator.tokenize(data_entry)
        inputs = [{"a": new_tokens, "b": all_tokens}]
        validator.generate(inputs, arg_fields=["a", "b"], result_field="result")
        assert not inputs[0]["result"]

        data_entry = "one two three"
        new_tokens = validator.tokenize(data_entry)
        inputs = [{"a": new_tokens, "b": all_tokens}]
        validator.generate(inputs, arg_fields=["a", "b"], result_field="result")
        assert inputs[0]["result"]
