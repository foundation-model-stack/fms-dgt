# Standard
import os
import shutil

# Third Party
import pandas as pd

# Local
from fms_dgt.blocks.postprocessors.dpk.fuzzy_dedup import FuzzyDedupPostprocessing


def test_dedup():

    test_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fuzzy_dedup_test"
    )

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

    fdedup = FuzzyDedupPostprocessing(
        type="fuzzy_dedup",
        name="test_fuzzy_dedup_postprocessor",
        processing_dir=test_root,
        restart=True,
    )
    df: pd.DataFrame = fdedup.generate(inp_df)

    assert (
        df.iloc[0]["output"][:54]
        == "mary had a little lamb, Its fleece was milky as snow. "
    )

    # Clean up test folder
    shutil.rmtree(test_root)
