# Standard
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union
import copy

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock, BaseBlockData
from fms_dgt.base.registry import register_block


@dataclass
class FlattenFieldData(BaseBlockData):
    to_flatten: Any
    flattened: Optional[Any] = None


@register_block("flatten_field")
class FlattenField(BaseBlock):
    """Flatten specified args"""

    DATA_TYPE = FlattenFieldData

    def execute(self, inputs: Iterable[FlattenFieldData]):
        outputs = []
        for x in inputs:
            to_flatten = x.to_flatten if type(x.to_flatten) == list else [x.to_flatten]
            for el in to_flatten:
                new_x = copy.deepcopy(x)
                new_x.flattened = el
                outputs.append(new_x)
        return outputs
