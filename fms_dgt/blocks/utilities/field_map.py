# Standard
from typing import Any, Dict, Iterable

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.registry import register_block


@register_block("field_map")
class FieldMapBlock(BaseBlock):
    """Map fields from one label to another"""

    def __init__(
        self,
        *args: Any,
        field_map: Dict,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not isinstance(field_map, dict):
            raise ValueError(f"field_map must be a dictionary")

        self._field_map = field_map

    def execute(self, inputs: Iterable[Dict], field_map: Dict = None):
        field_map = field_map or self._field_map
        for x in inputs:
            kvs = dict(x)
            for k, v in self._field_map.items():
                x[v] = kvs[k]
        return inputs
