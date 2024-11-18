# Standard
import os
import shutil

# Local
from fms_dgt.blocks.validators.rouge import RougeDedupValidator


def test_matches():

    tmp_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_cache")
    if os.path.exists(tmp_cache):
        shutil.rmtree(tmp_cache)

    all_data = [
        "I went to the store yesterday",
        "blue red yellow green",
    ]

    inputs = [
        {"input": "I went to the store"},
        {"input": "I went to the store"},
        {"input": "I went to the store yesterday"},
    ]
    validator = RougeDedupValidator(
        name="test_rouge_validator",
        threshold=1.0,
        datastore={"type": "default", "store_name": "rouge", "output_dir": tmp_cache},
    )
    validator(inputs, context=all_data)
    assert (
        inputs[0]["is_valid"]
        and not inputs[1]["is_valid"]
        and not inputs[2]["is_valid"]
    )

    shutil.rmtree(tmp_cache)
