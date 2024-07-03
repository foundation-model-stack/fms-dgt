# Standard
from typing import List
import json

# Third Party
import pytest

# First Party
from fms_dgt.base.instance import Instance
from fms_dgt.blocks.validators.rouge import RougeValidator


class TestRougeValidator:
    def test_matches(self):
        validator = RougeValidator("test_rouge_validator", {"threshold": 0.0})

        all_data = [
            "I went to the store",
            "I went to the store yesterday",
            "blue red yellow green",
        ]
        all_tokens = validator.tokenize(all_data)

        data_entry = "I went to the store"
        new_tokens = validator.tokenize(data_entry)
        args = [new_tokens, all_tokens]
        inputs = [Instance(args)]
        validator._threshold = 0.91
        validator.validate_batch(inputs)
        assert inputs[0].result

        data_entry = "one two three"
        new_tokens = validator.tokenize(data_entry)
        args = [new_tokens, all_tokens]
        inputs = [Instance(args)]
        validator._threshold = 0.0
        validator.validate_batch(inputs)
        assert not inputs[0].result
