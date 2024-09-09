# Standard
from typing import Any, List

# Local
from fms_dgt.base.instance import Instance
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.generators.llm import LMGenerator


@register_block("mock_llm")
class MockLlmGenerator(LMGenerator):
    """Mock LLM Generator"""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def generate_batch(self, requests: List[Instance], **kwargs) -> None:
        for req in requests:
            req.result = "Mock LLM result."

    def loglikelihood_batch(self, requests: List[Instance], **kwargs) -> None:
        # group requests by kwargs
        for req in requests:
            req.result = 1.0
