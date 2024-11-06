# Standard
from typing import Any
import sys

# Third Party
from data_processing.runtime.pure_python import PythonTransformLauncher
from data_processing.utils import ParamsUtils
from doc_quality_transform import (
    bad_word_filepath_cli_param,
    doc_content_column_cli_param,
    text_lang_cli_param,
)
from doc_quality_transform_python import DocQualityPythonTransformConfiguration

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.postprocessors import BaseLargeScaleProcessingBlock


@register_block("document_quality")
class DocumentQualityPostprocessing(BaseLargeScaleProcessingBlock):
    """Base Class for all Postprocessors"""

    def __init__(
        self,
        bad_word_filepath: str,
        text_lang: str = "en",
        doc_content_column: str = "contents",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.bad_word_filepath = bad_word_filepath
        self.text_lang = text_lang
        self.doc_content_column = doc_content_column

    def doc_quality(self, input_params: dict):

        local_conf = {
            "input_folder": self.input_dir,
            "output_folder": self.output_dir,
        }

        params = input_params | {
            # Data access. Only required parameters are specified
            "data_local_config": ParamsUtils.convert_to_ast(local_conf),
            # doc_quality params
            text_lang_cli_param: self.text_lang,
            doc_content_column_cli_param: self.doc_content_column,
            bad_word_filepath_cli_param: self.bad_word_filepath,
        }
        sys.argv = ParamsUtils.dict_to_req(d=params)
        # create launcher
        launcher = PythonTransformLauncher(
            runtime_config=DocQualityPythonTransformConfiguration()
        )
        # launch
        launcher.launch()

    def docquality_embeddable(
        self,
        runtime_code_location: str = "{'github': 'github', 'commit_hash': '12345', 'path': 'path'}",
        runtime_pipeline_id: str = "pipeline_id",
        runtime_job_id: str = "job_id",
    ):
        args = locals()
        args.pop("self")
        doc_quality_task = self.doc_quality(input_params=args)

    def _process(self, *args, **kwargs) -> None:
        self.docquality_embeddable()
