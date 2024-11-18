# Standard
from dataclasses import dataclass
from typing import Iterable, Optional

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock, BaseBlockData
from fms_dgt.base.registry import register_block


@dataclass
class PromptBuilderData(BaseBlockData):
    mapping: dict
    prompt: Optional[str] = None


@register_block("prompt_builder")
class PromptBuilder(BaseBlock):
    """Convert inputs into prompt"""

    DATA_TYPE: PromptBuilderData = PromptBuilderData

    def __init__(self, prompt_path: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        with open(prompt_path, "r") as f:
            self._prompt = f.read().strip()

    def execute(
        self,
        inputs: Iterable[PromptBuilderData],
    ):
        outputs = []
        for x in inputs:

            prompt = self._prompt
            for k, v in x.mapping.items():
                prompt = prompt.replace("{{" + k + "}}", v)

            x.prompt = prompt

            outputs.append(x)

        return outputs
