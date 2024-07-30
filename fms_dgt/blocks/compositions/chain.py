# Standard
from typing import Any, Dict, List

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.blocks import TYPE_KEY
from fms_dgt.utils import sdg_logger


@register_block("chain")
class BlockChain(BaseBlock):
    """Class for chain of blocks connected in a sequence, i.e., a BlockChain..."""

    def __init__(
        self,
        block_list: List[Dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if block_list is None:
            raise ValueError(f"'block_list' argument cannot be 'None'")

        for attr in ["arg_fields", "kwarg_fields", "result_field"]:
            if getattr(self, attr, False):
                sdg_logger.warn(
                    f"Attribute '{attr}' is set but it will not be used in block '{self.name}'"
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
            sdg_logger.info(f"Running block {block.name}")
            # initial block call will pass custom arg_fields / kwarg_fields / result_field
            block_data = block.generate(block_data)
        return block_data
