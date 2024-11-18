# Standard
from dataclasses import dataclass
from typing import Any, Iterable

# Local
from fms_dgt.base.block import BaseBlock, BaseBlockData
from fms_dgt.blocks.compositions.sequence import BlockSequence


def test_flatten():
    flatten_cfg1 = {
        "name": "f1",
        "type": "flatten_field",
    }
    flatten_cfg2 = {
        "name": "f2",
        "type": "flatten_field",
    }
    cfgs = [flatten_cfg1, flatten_cfg2]
    block_sequence = BlockSequence(
        blocks=cfgs,
        block_order=["f1", "f2"],
        input_maps=[None, {"flattened": "to_flatten"}],
    )
    data = [{"to_flatten": [[1, 2, 3], [4, 5, 6]]}]
    outputs = block_sequence(data)

    for i in range(1, 7):
        assert (
            outputs[i - 1]["flattened"] == i
        ), f"Expected {i} but got {outputs[i-1]['flattened']} at position {i-1}"


def test_args_kwargs():
    flatten_cfg1 = {
        "name": "f1",
        "type": "flatten_field",
    }
    flatten_cfg2 = {
        "name": "f2",
        "type": "flatten_field",
    }
    block_sequence = BlockSequence(
        blocks=[TestBlock(**flatten_cfg1), TestBlock(**flatten_cfg2)],
        block_order=["f1", "f2"],
        block_params=[
            {"args": [1], "kwargs": {"kwarg1": 2}},
            {"args": [3], "kwargs": {"kwarg1": 4}},
        ],
    )

    data = [{"input": [[1, 2, 3], [4, 5, 6]]}]
    outputs = block_sequence(data)
    expected = [{"input": (3, 4)}]
    assert (
        outputs == expected
    ), f"Incorrect output, expected {expected} but got {outputs}"

    block_params = [
        {"args": [5]},
        {"args": [7], "kwargs": {"kwarg1": 8}},
    ]
    outputs = block_sequence(data, block_params)
    expected = [{"input": (7, 8)}]
    assert (
        outputs == expected
    ), f"Incorrect output, expected {expected} but got {outputs}"


@dataclass
class TestBlockDataType(BaseBlockData):
    input: Any


class TestBlock(BaseBlock):
    """Flatten specified args"""

    DATA_TYPE = TestBlockDataType

    def execute(
        self, inputs: Iterable[TestBlockDataType], arg1: int, kwarg1: int = None
    ):
        outputs = []
        for x in inputs:
            x.input = (arg1, kwarg1) if arg1 or kwarg1 else x.input
            outputs.append(x)
        return outputs
