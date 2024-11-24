# Standard
from typing import Any, Dict, List, Union

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.constants import DATASET_TYPE, NAME_KEY, TYPE_KEY
from fms_dgt.utils import sdg_logger


@register_block("sequence")
class BlockSequence(BaseBlock):
    """Class for sequence of blocks connected in a sequence"""

    def __init__(
        self,
        blocks: List[Union[Dict, BaseBlock]],
        block_sequence: List[Dict],
        *args,
        **kwargs: Any,
    ) -> None:
        """Class for specifying a sequence of blocks, where the outputs of one block are immediately passed as input to the next block

        Args:
            blocks (List[Union[Dict, BaseBlock]]): List of blocks to initialize and use within the chain of blocks.
            block_sequence (List[str]): The order in which to call blocks and any args / kwargs
        """
        super().__init__(*args, **kwargs)

        if len(
            set(
                [
                    (block.name if isinstance(block, BaseBlock) else block[NAME_KEY])
                    for block in blocks
                ]
            )
        ) != len(blocks):
            raise ValueError(f"Duplicate block detected in blocks list [{blocks}]")

        self._block_sequence = block_sequence

        self._blocks_map: Dict[str, BaseBlock] = {
            (block.name if isinstance(block, BaseBlock) else block[NAME_KEY]): (
                block
                if isinstance(block, BaseBlock)
                else get_block(block_name=block[TYPE_KEY], **block)
            )
            for block in blocks
        }
        self._blocks = list(self._blocks_map.values())

    @property
    def blocks(self):
        return self._blocks

    def __call__(self, *args, **kwargs) -> DATASET_TYPE:
        return self.execute(*args, **kwargs)

    def execute(
        self, inputs: DATASET_TYPE, block_sequence: List[Dict] = None
    ) -> DATASET_TYPE:
        """Sequence of blocks chained together

        Args:
            inputs (DATASET_TYPE): Data to process with blocks.

        Kwargs:
            block_sequence (List[Dict], optional): Override for self._block_sequence, i.e., block_sequence specified in the __init__.

        Returns:
            DATASET_TYPE: Data that has been passed through all blocks.
        """

        block_sequence = block_sequence or self._block_sequence

        block_data = inputs
        for block_info in block_sequence:
            block_info = dict(block_info)
            block_name = block_info.pop(NAME_KEY)
            sdg_logger.info("Running block %s", block_name)
            block = next(iter([b for b in self.blocks if b.name == block_name]))
            # execute processing
            block_data = block(block_data, **block_info)

        return block_data


def validate_block_sequence(block_list: List[Dict]):
    for block in block_list:
        if not isinstance(block, dict):
            raise ValueError("Block in block sequence must be a dictionary")
        if block.get(NAME_KEY) is None:
            raise ValueError(f"Must specify {NAME_KEY} in block {block}")
