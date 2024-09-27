# Standard
import os
import shutil

# Third Party
import pandas as pd

# Local
from fms_dgt.blocks.postprocessors.dpk.code_quality import CodeQualityPostprocessing


def test_codequality():

    test_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "codequality_test"
    )

    test_data = {
        "output": [
            "thislist = [1, 3, 5] for x in thislist: print(x)",
            "a=1 \n b=2 \n c=3 \n x=1 \n y=2 \n z=3",
            '<?xml version="1.0"?> <note> <to>Tove</to> <from>Jani</from> '
            "<heading>Reminder</heading> <body>Don't forget me this weekend!</body> </note>",
        ],
        "language": ["python", "python", "xml"],
    }
    inp_df = pd.DataFrame(data=test_data)

    codequality = CodeQualityPostprocessing(
        type="code_quality",
        name="test_codequality_postprocessor",
        processing_dir=test_root,
        hf_token="",
        contents_column_name="output",
        language_column_name="language",
        tokenizer="codeparrot/codeparrot",
        restart=True,
    )

    df: pd.DataFrame = codequality.generate(inp_df)

    assert df.iloc[0]["has_few_assignments"] == True
    assert df.iloc[1]["has_few_assignments"] == False
    assert df.iloc[1]["has_no_keywords"] == True
    assert df.iloc[2]["is_xml"] == True

    # Clean up test folder
    shutil.rmtree(test_root)
