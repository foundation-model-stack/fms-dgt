# Standard
import os
import shutil

# Third Party
import pandas as pd

# Local
from fms_dgt.blocks.postprocessors.dpk.fuzzy_dedup import FuzzyDedupPostprocessing
from fms_dgt.datastores.default import DefaultDatastore


def test_dedup():

    tmp_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_cache")
    test_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fuzzy_dedup_test"
    )
    for d in [tmp_cache, test_root]:
        if os.path.exists(d):
            shutil.rmtree(d)

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

    from_ds = DefaultDatastore(
        output_dir=os.path.join(tmp_cache, "from"), store_name="input"
    )
    to_ds = DefaultDatastore(
        output_dir=os.path.join(tmp_cache, "to"), store_name="output"
    )
    from_ds.save_data(inp_df)

    fdedup = FuzzyDedupPostprocessing(
        type="fuzzy_dedup",
        name="test_fuzzy_dedup_postprocessor",
        processing_dir=test_root,
        restart=True,
    )
    fdedup([("mock_task", from_ds, to_ds)])

    df = pd.DataFrame(to_ds.load_data())

    assert (
        df.iloc[0]["output"][:54]
        == "mary had a little lamb, Its fleece was milky as snow. "
    )

    # clean up test folders
    for d in [tmp_cache, test_root]:
        if os.path.exists(d):
            shutil.rmtree(d)
