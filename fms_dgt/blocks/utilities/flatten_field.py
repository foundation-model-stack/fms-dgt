# Standard
from typing import Dict, List, Optional, Union
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
        fields: Optional[Union[List, Dict]] = None,
        result_field: Optional[str] = None,
    ):
        fields = fields or self._fields or []

        assert len(fields) == 1, f"{self.__class__.__name__} can only have 1 field!"

        outputs = []
        for x in inputs:
            to_flatten = list(self.get_args_kwargs(x, fields).values())[0]
            to_flatten = to_flatten if type(to_flatten) == list else [to_flatten]
            for el in to_flatten:
                outputs.append(copy.copy(x))
                self.write_result(outputs[-1], el, result_field)

        return outputs
