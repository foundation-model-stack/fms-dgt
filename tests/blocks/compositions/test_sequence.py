# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock
from fms_dgt.blocks.compositions.sequence import BlockSequence


def test_flatten():
    flatten_cfg1 = {
        "name": "f1",
        "type": "flatten_field",
        "arg_fields": ["arg"],
        "result_field": "arg",
    }
    flatten_cfg2 = {
        "name": "f2",
        "type": "flatten_field",
        "arg_fields": ["arg"],
        "result_field": "arg",
    }
    cfgs = [flatten_cfg1, flatten_cfg2]
    block_sequence = BlockSequence(
        blocks=cfgs,
        block_order=["f1", "f2"],
    )
    data = [{"arg": [[1, 2, 3], [4, 5, 6]]}]
    outputs = block_sequence(data)
    for i in range(1, 7):
        assert (
            outputs[i - 1]["arg"] == i
        ), f"Expected {i} but got {outputs[i-1]['arg']} at position {i-1}"


def test_args_kwargs():
    flatten_cfg1 = {
        "name": "f1",
        "type": "flatten_field",
        "arg_fields": ["arg"],
        "result_field": "arg",
    }
    flatten_cfg2 = {
        "name": "f2",
        "type": "flatten_field",
        "arg_fields": ["arg"],
        "result_field": "arg",
    }
    block_sequence = BlockSequence(
        blocks=[TestBlock(**flatten_cfg1), TestBlock(**flatten_cfg2)],
        block_order=["f1", "f2"],
        block_params=[
            {"args": [1], "kwargs": {"kwarg1": 2}},
            {"args": [3], "kwargs": {"kwarg1": 4}},
        ],
    )
    data = [{"arg": [[1, 2, 3], [4, 5, 6]]}]
    outputs = block_sequence(data)

    expected = [{"arg": (3, 4)}]
    assert (
        outputs == expected
    ), f"Incorrect output, expected {expected} but got {outputs}"

    block_params = [
        {"args": [5]},
        {"args": [7], "kwargs": {"kwarg1": 8}},
    ]
    outputs = block_sequence(data, block_params)
    expected = [{"arg": (7, 8)}]
    assert (
        outputs == expected
    ), f"Incorrect output, expected {expected} but got {outputs}"


class TestBlock(BaseBlock):
    """Flatten specified args"""

    def execute(self, inputs: DATASET_TYPE, arg1: int, kwarg1: int = None):
        outputs = []
        for x in inputs:
            inp_args, _ = self.get_args_kwargs(x, self._arg_fields, self._kwarg_fields)
            to_write = (arg1, kwarg1) if arg1 or kwarg1 else inp_args
            self.write_result(x, to_write)
            outputs.append(x)
        return outputs
