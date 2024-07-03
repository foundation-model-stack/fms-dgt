# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.block import BaseGeneratorBlock
from fms_dgt.base.instance import Instance
from fms_dgt.base.registry import register_block


@register_block("template_generator")
class TemplateGenerator(BaseGeneratorBlock):
    """Base Class for all Generators"""

    def __init__(self, name: str, config: Dict, **kwargs: Any) -> None:
        super().__init__(name, config, **kwargs)

    def __call__(self, inputs: List[Instance], **kwargs: Any) -> None:
        raise NotImplementedError
