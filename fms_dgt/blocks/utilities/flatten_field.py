# Standard
from typing import List, Optional
import copy

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.base.registry import register_block


@register_block("flatten_field")
class FlattenField(BaseBlock):
    """Flatten specified args"""

    def execute(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
    ):
        arg_fields = arg_fields or self._arg_fields or []

        assert (
            len(arg_fields) == 1
        ), f"{self.__class__.__name__} can only have 1 arg field!"

        outputs = []
        for x in inputs:
            inp_args, _ = self.get_args_kwargs(x, arg_fields, kwarg_fields)
            to_flatten = inp_args[0] if type(inp_args[0]) == list else [inp_args[0]]
            for el in to_flatten:
                outputs.append(copy.copy(x))
                self.write_result(outputs[-1], el, result_field)

        return outputs
