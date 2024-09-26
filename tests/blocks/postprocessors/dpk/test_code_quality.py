# Local
from fms_dgt.blocks.postprocessors.dpk.code_quality import CodeQualityPostprocessing


class TestCodeQualityPostprocessor:
    def test_codequality(self):
        # Standard
        from pathlib import Path
        import os

        # Third Party
        import pandas as pd

        cwd = Path.cwd()
        test_root = cwd / "codequality_test"

        os.makedirs(test_root, exist_ok=True)
        input_folder = os.path.join(test_root, "test_input")
        os.makedirs(input_folder, exist_ok=True)
        output_folder = os.path.join(test_root, "test_output")
        os.makedirs(output_folder, exist_ok=True)

        test_data = {
            "output": [
                "thislist = [1, 3, 5] for x in thislist: print(x)",
                "a=1 \n b=2 \n c=3 \n x=1 \n y=2 \n z=3",
                '<?xml version="1.0"?> <note> <to>Tove</to> <from>Jani</from> '
                "<heading>Reminder</heading> <body>Don't forget me this weekend!</body> </note>",
            ],
            "language": ["python", "python", "xml"],
        }
        df = pd.DataFrame(data=test_data)
        df.to_parquet(os.path.join(input_folder, "test.parquet"))

        codequality = CodeQualityPostprocessing(
            name="test_codequality_postprocessor",
            input_folder_path=input_folder,
            output_folder_path=output_folder,
            hf_token="",
            contents_column_name="output",
            language_column_name="language",
            tokenizer="codeparrot/codeparrot",
        )
        codequality.generate(inputs=None)
        # Load processed data
        df = pd.read_parquet(os.path.join(output_folder, "test.parquet"))

        assert df.iloc[0]["has_few_assignments"] == True
        assert df.iloc[1]["has_few_assignments"] == False
        assert df.iloc[1]["has_no_keywords"] == True
        assert df.iloc[2]["is_xml"] == True

        # Clean up test folder
        # Standard
        import shutil

        shutil.rmtree(test_root)


simple = TestCodeQualityPostprocessor()
simple.test_codequality()
