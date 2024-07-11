# Standard
from typing import Any, Dict, List, Optional, Union

# Third Party
from datasets import Dataset
from pandas import DataFrame

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.registry import register_block


@register_block("template_validator")
class TemplateValidator(BaseValidatorBlock):
    """Base Class for all Validators"""

    def __init__(self, name: str, config: Dict) -> None:
        super().__init__(name, config)

    def _validate(self, *args, **kwargs) -> bool:
        """Return True if valid and False otherwise"""
        return True
