# Standard
from typing import Dict

# Local
from fms_sdg.base.registry import register_resource
from fms_sdg.resources.api_key import ApiKeyResource


@register_resource("openai")
class OpenAIKeyResource(ApiKeyResource):

    OPEN_AI_CALL_LIMIT = 10

    def __init__(
        self, key_name: str = "OPENAI_API_KEY", call_limit: int = OPEN_AI_CALL_LIMIT
    ):
        super().__init__(key_name, call_limit)
