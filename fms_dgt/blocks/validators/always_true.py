# Standard
from dataclasses import dataclass
from typing import Any

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.validators import BaseValidatorBlock, BaseValidatorBlockData


@dataclass
class AlwaysTrueBlockData(BaseValidatorBlockData):
    input: Any


@register_block("always_true")
class AlwaysTrueValidator(BaseValidatorBlock):
    """Class for placeholder validator that always returns true"""

    DATA_TYPE = AlwaysTrueBlockData

    def _validate(self, input: AlwaysTrueBlockData) -> bool:
        return True
