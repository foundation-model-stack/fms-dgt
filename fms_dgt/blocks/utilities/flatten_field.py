# Standard
from typing import Any, Dict, List, Optional, Union
import copy

# Third Party
from datasets import Dataset
from pandas import DataFrame

# Local
from fms_dgt.base.block import BaseUtilityBlock
from fms_dgt.base.registry import register_block


@register_block("flatten_field")
class FlattenFieldBlock(BaseUtilityBlock):
    """Flatten specified args"""

    def __call__(
        self,
        inputs: Union[List[Dict], DataFrame, Dataset],
        *args: Any,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        arg_fields = arg_fields if arg_fields is not None else self._arg_fields
        if arg_fields is None:
            arg_fields = []

        assert (
            len(arg_fields) == 1
        ), f"{self.__class__.__name__} can only have 1 arg field!"

        outputs = []
        for x in inputs:
            inp_args, inp_kwargs = self.get_args_kwargs(x, arg_fields, kwarg_fields)
            to_flatten = inp_args[0] if type(inp_args[0]) == list else [inp_args[0]]

            # remove flattened attribute
            x_copy = copy.copy(x)
            delattr(x_copy, arg_fields[0])

            for el in to_flatten:
                outputs.append(copy.copy(x_copy))
                delattr(outputs[-1], arg_fields[0])
                self.write_result(outputs[-1], el, result_field)

        return outputs
