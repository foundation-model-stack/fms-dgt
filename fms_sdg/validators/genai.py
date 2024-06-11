# Standard
from typing import Any, Dict, List

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_validator
from fms_sdg.base.validator import BaseValidator
from fms_sdg.generators.genai import GenAIGenerator


@register_validator("genai")
class GenAIValidator(GenAIGenerator, BaseValidator):
    """GenAI Validator"""

    def __init__(self, name: str, config: Dict, **kwargs: Any):
        super().__init__(name, config, **kwargs)

    def validate_batch(self, inputs: List[Instance], **kwargs: Any) -> None:
        pass
