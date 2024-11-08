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
        block_order: List[str],
        *args,
        block_params: List[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Class for specifying a sequence of blocks, where the outputs of one block are immediately passed as input to the next block

        Args:
            blocks (List[Union[Dict, BaseBlock]]): List of blocks to initialize and use within the chain of blocks.
            block_order (List[str]): The order in which to call blocks.

        Kwargs:
            block_params (List[Dict], optional): A list of entries of the form [{'args': [arg_val1, arg_val2, ...], 'kwargs': {'kwarg1' : kwarg_val1, 'kwarg2' : kwarg_val2, ...}}, ...].
                This list will be zipped together with the blocks specified in [block_order]. If this list is specified, it must be the SAME length as [block_order].
        """
        super().__init__(*args, **kwargs)

        for attr in [self._arg_fields, self._kwarg_fields, self._result_field]:
            if attr is not None:
                sdg_logger.warning(
                    "Field attribute is set but it will not be used in block '%s'",
                    self.name,
                )

        if len(
            set(
                [
                    (block.name if isinstance(block, BaseBlock) else block[NAME_KEY])
                    for block in blocks
                ]
            )
        ) != len(blocks):
            raise ValueError(f"Duplicate block detected in blocks list [{blocks}]")

        self._block_params = block_params
        self._block_order = block_order

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

    def _get_block_params(self, block_params: List[Dict]):

        block_params = (
            block_params or self._block_params or [dict() for _ in self._block_order]
        )

        if len(block_params) != len(self._block_order):
            raise ValueError(
                f"When provided, [blocks_args_kwargs] should be the same length as [block_list]"
            )

        for d in block_params:
            for k in d:
                if k not in ["args", "kwargs"]:
                    raise ValueError(
                        f"All keys to dictionary in [block_params] must be either [args, kwargs], instead got [{k}] for dictionary [{d}]"
                    )
                for v, t in [("args", list), ("kwargs", dict)]:
                    if d.get(v) and not isinstance(d.get(v), t):
                        raise ValueError(
                            f"Expected {v} to be {t}, instead got {type(d[v])}"
                        )

        return block_params

    def execute(
        self, inputs: DATASET_TYPE, block_params: List[Dict] = None
    ) -> DATASET_TYPE:
        """_summary_

        Args:
            inputs (DATASET_TYPE): Data to process with blocks.
            block_params (List[Dict], optional): Override for self._block_params, i.e., block_params specified in the __init__.

        Returns:
            DATASET_TYPE: Data that has been passed through all blocks.
        """
        block_params = self._get_block_params(block_params)

        block_data = inputs
        for block_name, args_kwargs in zip(self._block_order, block_params):

            sdg_logger.info("Running block %s", block_name)

            block = self._blocks_map[block_name]

            args = args_kwargs.get("args", [])
            kwargs = args_kwargs.get("kwargs", dict())

            block_data = block(block_data, *args, **kwargs)

        return block_data
