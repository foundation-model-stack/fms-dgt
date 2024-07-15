# Standard
from typing import List, Optional
import copy

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.base.registry import register_block


@register_block("prompt_builder")
class PromptBuilder(BaseBlock):
    """Flatten specified args"""

    def generate(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
    ):
        outputs = []
        for x in inputs:
            inp_args, _ = self.get_args_kwargs(x, arg_fields, kwarg_fields)
            to_flatten = inp_args[0] if type(inp_args[0]) == list else [inp_args[0]]

            # remove flattened attribute
            x_copy = copy.copy(x)
            delattr(x_copy, arg_fields[0])

            for el in to_flatten:
                outputs.append(copy.copy(x_copy))
                delattr(outputs[-1], arg_fields[0])
                self.write_result(outputs[-1], el, result_field)

        return outputs
