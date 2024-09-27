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

    data = [{"a": "1", "b": "2", "c": "3"} for _ in range(100000)]

    proc_data = block.generate(data, arg_fields=["b"])

    shutil.rmtree(_B_PATH)

    assert data and data != proc_data, f"Expected data to be deduplicated"
