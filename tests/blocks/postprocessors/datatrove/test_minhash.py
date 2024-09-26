# Standard
import os
import shutil

# Local
from fms_dgt.blocks.postprocessors.datatrove.minhash_dedup import MinHashDatatrove

_B_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_block")


def test_minhash():

    block = MinHashDatatrove(
        type="default",
        name="dstore",
        folder_path=_B_PATH,
        restart=True,
    )

    data = [
        {"a": "1", "b": "2", "c": "3"},
        {"a": "1", "b": "0", "c": "3"},
        {"a": "1", "b": "2", "c": "3"},
    ]

    proc_data = block.generate(data, arg_fields=["b"])

    assert data == proc_data, f"Data mismatch, expected {data} but got {proc_data}"

    shutil.rmtree(_B_PATH)
