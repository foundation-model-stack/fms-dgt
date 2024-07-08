# Standard
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from pandas import DataFrame

# Local
from fms_dgt.base.block import BLOCK_INPUT_TYPE, BaseGeneratorBlock
from fms_dgt.base.registry import register_block


@register_block("template_generator")
class TemplateGenerator(BaseGeneratorBlock):
    """Base Class for all Generators"""

    def __init__(self, name: str, config: Dict, **kwargs: Any) -> None:
        super().__init__(name, config, **kwargs)

    def __call__(
        self,
        inputs: BLOCK_INPUT_TYPE,
        *args: Any,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError
