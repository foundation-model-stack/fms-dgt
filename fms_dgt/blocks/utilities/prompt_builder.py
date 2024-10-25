# Standard
from typing import List, Optional

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.base.registry import register_block


@register_block("prompt_builder")
class PromptBuilder(BaseBlock):
    """Convert inputs into prompt"""

    def __init__(self, prompt_path: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        with open(prompt_path, "r") as f:
            self._prompt = f.read().strip()

    def execute(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
    ):
        outputs = []
        for x in inputs:
            _, inp_kwargs = self.get_args_kwargs(x, arg_fields, kwarg_fields)

            prompt = self._prompt
            for k, v in inp_kwargs.items():
                prompt = prompt.replace("{{" + k + "}}", v)

            outputs.append(x)

            self.write_result(outputs[-1], prompt, result_field)

        return outputs
