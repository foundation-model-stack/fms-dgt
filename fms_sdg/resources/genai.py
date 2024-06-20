# Standard
from typing import Dict

# Local
from fms_sdg.base.registry import register_resource
from fms_sdg.resources.api import ApiKeyResource


@register_resource("genai")
class GenAIKeyResource(ApiKeyResource):

    GENAI_CALL_LIMIT = 10

    def __init__(self, key_name: str = "GENAI_KEY", call_limit: int = GENAI_CALL_LIMIT):
        super().__init__(key_name, call_limit)
