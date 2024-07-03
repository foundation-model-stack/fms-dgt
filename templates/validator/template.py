# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.instance import Instance
from fms_dgt.base.registry import register_block


@register_block("template_validator")
class TemplateValidator(BaseValidatorBlock):
    """Base Class for all Validators"""

    def __init__(self, name: str, config: Dict) -> None:
        super().__init__(name, config)

    def __call__(self, inputs: List[Instance], **kwargs: Any) -> None:
        """Takes in a list of Instance objects (each containing their own arg / kwargs)"""
        for x in inputs:
            x.result = self._validate(*x.args, **x.kwargs)

    def _validate(self, *args, **kwargs) -> bool:
        """Return True if valid and False otherwise"""
        return True
