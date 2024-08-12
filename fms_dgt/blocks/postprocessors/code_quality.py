# Local
from fms_dgt.base.block import BaseBlock, DATASET_TYPE
from fms_dgt.base.registry import register_block
from typing import Any, Dict, List, Optional, Union

@register_block("code_quality")
class CodeQualityPostprocessing(BaseBlock):
    """Base Class for all Postprocessors"""

    def __init__(self,
                 input_folder_path: str,
                 output_folder_path: str,
                 hf_token: str,
                 contents_column_name: str = "contents",
                 language_column_name: str = "language",
                 tokenizer: str = "codeparrot/codeparrot",
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.contents_column_name = contents_column_name
        self.language_column_name = language_column_name
        self.tokenizer = tokenizer
        self.hf_token = hf_token

    def code_quality(self, input_params: dict):
        import sys

        from code_quality_transform import CodeQualityTransformConfiguration
        from data_processing.runtime.pure_python import PythonTransformLauncher
        from data_processing.utils import ParamsUtils

        local_conf = {
            "input_folder": self.input_folder_path,
            "output_folder": self.output_folder_path,
        }

        params = input_params | {
            # Data access. Only required parameters are specified
            "data_local_config": ParamsUtils.convert_to_ast(local_conf),
            "cq_contents_column_name": self.contents_column_name,
            "cq_language_column_name": self.language_column_name,
            "cq_tokenizer": self.tokenizer,
            "cq_hf_token": self.hf_token,
        }
        sys.argv = ParamsUtils.dict_to_req(d=params)
        # create launcher
        launcher = PythonTransformLauncher(runtime_config=CodeQualityTransformConfiguration())
        # launch
        launcher.launch()

    def codequality_embeddable(self,
            runtime_code_location: str = "{'github': 'github', 'commit_hash': '12345', 'path': 'path'}",
            runtime_pipeline_id: str = "pipeline_id",
            runtime_job_id: str = "job_id",
    ):
        args = locals()
        args.pop("self")
        code_quality_task = self.code_quality(input_params=args)

    def generate(self, inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,):

        self._postprocess()

        return self.output_folder_path

    def _postprocess(self) -> bool:
        self.codequality_embeddable()
        return True

