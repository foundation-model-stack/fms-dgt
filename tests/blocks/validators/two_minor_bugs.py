# Local
from fms_dgt.blocks.validators.rouge import RougeDedupValidator


class TestRougeValidator:
    def test_crash_on_None(self):
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=0.0)

        all_data = [
            "I went to the store yesterday",
            "blue red yellow green",
        ]

        inputs = [{"a": "I went to the store"}, {"a": "I went to the store yesterday"}]
        # validator._threshold = 0.91 # crash on default None
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        assert inputs[0]["result"] and not inputs[1]["result"]

    def test_incorrect_max_fmeasure(self):
        validator = RougeDedupValidator(name="test_rouge_validator", threshold=0.0)

        all_data = [
            # "I went to the store yesterday x y z",
            "I went to the store yesterday x",
            "I went to the store y w z y",
            "blue red yellow green",
        ]

        inputs = [{"a": "I went to the store x y z"}]
        # It is difficult to be sure I compute the rouge scores here,
        # so you have to use my modified rouge.py to see that
        # the input has a max fmeasure of 0.82,
        # so it should have a result of False with threshold 0.81,
        # but since the max is taken over Score tuples,
        # the incorrect max fmeasure it computed.
        # pytest -rx -rP tests/blocks/validators/two_minor_bugs.py
        validator._threshold = 0.81
        validator.generate(
            inputs, context=all_data, arg_fields=["a"], result_field="result"
        )
        assert not inputs[0]["result"]
