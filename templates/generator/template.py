# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.generator import BaseGenerator
from fms_dgt.base.instance import Instance
from fms_dgt.base.registry import register_generator


@register_generator("template_generator")
class TemplateGenerator(BaseGenerator):
    """Base Class for all Generators"""

    def __init__(self, name: str, config: Dict, **kwargs: Any) -> None:
        super().__init__(name, config, **kwargs)

    def generate_batch(self, inputs: List[Instance], **kwargs: Any) -> None:
        raise NotImplementedError
