# Standard
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from pandas import DataFrame

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.instance import Instance
from fms_dgt.base.registry import register_block


@register_block("template_validator")
class TemplateValidator(BaseValidatorBlock):
    """Base Class for all Validators"""

    def __init__(self, name: str, config: Dict) -> None:
        super().__init__(name, config)

    def __call__(
        self,
        inputs: Union[List[Dict], DataFrame, Dataset],
        *args: Any,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        return super().__call__(
            inputs,
            *args,
            arg_fields=arg_fields,
            kwarg_fields=kwarg_fields,
            result_field=result_field,
            **kwargs,
        )

    def _validate(self, *args, **kwargs) -> bool:
        """Return True if valid and False otherwise"""
        return True
