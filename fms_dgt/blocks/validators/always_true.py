# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.validators import BaseValidatorBlock


@register_block("always_true")
class AlwaysTrueValidator(BaseValidatorBlock):
    """Class for placeholder validator that always returns true"""

    def _validate(self, *args, **kwargs) -> bool:
        return True
