# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.blocks import TYPE_KEY
from fms_dgt.utils import sdg_logger


@register_block("sequence")
class BlockSequence(BaseBlock):
    """Class for sequence of blocks connected in a sequence..."""

    def __init__(
        self,
        block_list: List[Dict],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for attr in [self._arg_fields, self._kwarg_fields, self._result_field]:
            if attr is not None:
                sdg_logger.warn(
                    "Field attribute is set but it will not be used in block '%s'",
                    self.name,
                )

        self._blocks: List[BaseBlock] = [
            get_block(block_name=block_info[TYPE_KEY], **block_info)
            for block_info in block_list
        ]

    @property
    def blocks(self):
        return self._blocks

    def generate(self, inputs: DATASET_TYPE):
        block_data = inputs
        for block in self.blocks:
            sdg_logger.info("Running block %s", block.name)
            # initial block call will pass custom arg_fields / kwarg_fields / result_field
            block_data = block.generate(block_data)
        return block_data
