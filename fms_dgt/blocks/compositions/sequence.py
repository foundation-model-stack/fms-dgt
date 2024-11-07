# Standard
from typing import Any, Dict, List, Union

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.registry import get_block, register_block
from fms_dgt.constants import DATASET_TYPE, TYPE_KEY
from fms_dgt.utils import sdg_logger


@register_block("sequence")
class BlockSequence(BaseBlock):
    """Class for sequence of blocks connected in a sequence..."""

    def __init__(
        self,
        block_list: List[Union[Dict, BaseBlock]],
        block_args_kwargs: List[Dict] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        for attr in [self._arg_fields, self._kwarg_fields, self._result_field]:
            if attr is not None:
                sdg_logger.warning(
                    "Field attribute is set but it will not be used in block '%s'",
                    self.name,
                )

        self._block_args_kwargs = block_args_kwargs

        self._blocks: List[BaseBlock] = [
            (
                get_block(block_name=block[TYPE_KEY], **block)
                if isinstance(block, dict)
                else block
            )
            for block in block_list
        ]

    @property
    def blocks(self):
        return self._blocks

    def _get_block_args_kwargs(self, block_args_kwargs: List[Dict]):

        block_args_kwargs = (
            block_args_kwargs
            or self._block_args_kwargs
            or [dict() for _ in self.blocks]
        )

        if len(block_args_kwargs) != len(self.blocks):
            raise ValueError(
                f"When provided, [blocks_args_kwargs] should be the same length as [block_list]"
            )

        for d in block_args_kwargs:
            for k in d:
                if k not in ["args", "kwargs"]:
                    raise ValueError(
                        f"All keys to dictionary in [block_args_kwargs] must be either [args, kwargs], instead got [{k}] for dictionary [{d}]"
                    )
                for v, t in [("args", list), ("kwargs", dict)]:
                    if not isinstance(d[v], t):
                        raise ValueError(
                            f"Expected {v} to be {t}, instead got {type(d[v])}"
                        )

        return block_args_kwargs

    def execute(self, inputs: DATASET_TYPE, block_args_kwargs: List[Dict] = None):

        block_args_kwargs = self._get_block_args_kwargs(block_args_kwargs)

        block_data = inputs
        for block, args_kwargs in zip(self.blocks, block_args_kwargs):
            sdg_logger.info("Running block %s", block.name)
            # initial block call will pass custom arg_fields / kwarg_fields / result_field
            args, kwargs = args_kwargs.get("args", []), args_kwargs.get(
                "kwargs", dict()
            )
            block_data = block(block_data, *args, **kwargs)
        return block_data
