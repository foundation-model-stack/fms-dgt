# Local
from fms_dgt.base.registry import register_resource
from fms_dgt.resources.api import ApiKeyResource


@register_resource("openai")
class OpenAIKeyResource(ApiKeyResource):

    OPEN_AI_CALL_LIMIT = 10

    def __init__(
        self, key_name: str = "OPENAI_API_KEY", call_limit: int = OPEN_AI_CALL_LIMIT
    ):
        super().__init__(key_name, call_limit)
