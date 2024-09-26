# Local
from fms_dgt.blocks.postprocessors.dpk.document_quality import (
    DocumentQualityPostprocessing,
)


class TestDocumentQualityPostprocessor:
    def test_docquality(self):
        # Standard
        from pathlib import Path
        import os

        # Third Party
        import pandas as pd

        cwd = Path.cwd()
        test_root = cwd / "docquality_test"
        bad_word_filepath = cwd / "test_data/docquality/ldnoobw/en"

        os.makedirs(test_root, exist_ok=True)
        input_folder = os.path.join(test_root, "test_input")
        os.makedirs(input_folder, exist_ok=True)
        output_folder = os.path.join(test_root, "test_output")
        os.makedirs(output_folder, exist_ok=True)

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
        df = pd.DataFrame(data=test_data)
        df.to_parquet(os.path.join(input_folder, "test.parquet"))

        docquality = DocumentQualityPostprocessing(
            name="test_docquality_postprocessor",
            input_folder_path=input_folder,
            output_folder_path=output_folder,
            bad_word_filepath=bad_word_filepath,
            text_lang="en",
            doc_content_column="output",
        )
        docquality.generate(inputs=None)
        # Load processed data
        df = pd.read_parquet(os.path.join(output_folder, "test.parquet"))

        assert df.iloc[0]["docq_contain_bad_word"] == False
        assert df.iloc[4]["docq_contain_bad_word"] == True

        # Clean up test folder
        # Standard
        import shutil

        shutil.rmtree(test_root)


simple = TestDocumentQualityPostprocessor()
simple.test_docquality()
