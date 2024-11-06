# Standard
import os
import shutil

# Third Party
import pandas as pd

# Local
from fms_dgt.blocks.postprocessors.datatrove.minhash_dedup import MinHashDatatrove


def test_minhash():

    test_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minhash_test")
    if os.path.exists(test_root):
        shutil.rmtree(test_root)

    test_data = {
        "output": [
            "mary had a little lamb, Its fleece was milky as snow. "
            "And everywhere that Mary went, The lamb was sure to go. He followed her "
            "to school one day, that was against the rule. It made the children laugh and play. "
            "To see a lamb at school.",
            "Mary had a little lamb, Its fleece was white as snow. "
            "And everywhere that Mary went, The lamb was sure to go. He followed her "
            "to school one day, that was against the rule. It made the children laugh and play. "
            "To see a lamb at school.",
            "London Bridge is falling down, falling down, falling down. "
            "London Bridge is falling down, My fair lady.",
            "The wheels on the bus go round and round, round and round, round and round. "
            "The wheels on the bus go round and round, All through the town. ",
        ]
    }
    inp_df = pd.DataFrame(data=test_data)

    minhash = MinHashDatatrove(
        type="default",
        name="test_minhash_postprocessor",
        processing_dir=test_root,
        text_key="output",
        restart=True,
    )

    out_df = minhash(inp_df.to_dict(orient="records"))

    df = pd.DataFrame(out_df)

    assert len(df) < len(set(test_data["output"]))

    # clean up test folders
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
