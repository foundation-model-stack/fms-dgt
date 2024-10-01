# Standard
import os
import shutil

# Third Party
import pandas as pd

# Local
from fms_dgt.blocks.postprocessors.dpk.document_quality import (
    DocumentQualityPostprocessing,
)
from fms_dgt.datastores.default import DefaultDatastore


def test_docquality():

    tmp_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_cache")
    test_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "docquality_test"
    )
    for d in [tmp_cache, test_root]:
        if os.path.exists(d):
            shutil.rmtree(d)

    bad_word_filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "test_data",
        "docquality",
        "ldnoobw",
        "en",
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
            "This is a document containing xxx material.",
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

    docquality = DocumentQualityPostprocessing(
        type="document_quality",
        name="test_docquality_postprocessor",
        processing_dir=test_root,
        bad_word_filepath=bad_word_filepath,
        text_lang="en",
        doc_content_column="output",
        restart=True,
    )
    docquality.generate([("mock_task", from_ds, to_ds)])

    df = pd.DataFrame(to_ds.load_data())

    assert df.iloc[0]["docq_contain_bad_word"] == False
    assert df.iloc[4]["docq_contain_bad_word"] == True

    # clean up test folders
    for d in [tmp_cache, test_root]:
        if os.path.exists(d):
            shutil.rmtree(d)
