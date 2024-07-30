# Local
from fms_dgt.blocks.compositions.chain import BlockChain


class TestBlockChain:
    def test_flatten(self):
        flatten_cfg1 = {
            "type": "flatten_field",
            "arg_fields": ["arg"],
            "result_field": "arg",
        }
        flatten_cfg2 = {
            "type": "flatten_field",
            "arg_fields": ["arg"],
            "result_field": "arg",
        }
        cfgs = [flatten_cfg1, flatten_cfg2]
        block_chain = BlockChain(cfgs)
        data = [{"arg": [[1, 2, 3], [4, 5, 6]]}]
        outputs = block_chain.generate(data)
        for i in range(1, 7):
            assert (
                outputs[i - 1]["arg"] == i
            ), f"Expected {i} but got {outputs[i-1]['arg']} at position {i-1}"
