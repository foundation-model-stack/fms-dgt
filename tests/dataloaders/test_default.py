# Local
from fms_dgt.dataloaders.default import DefaultDataloader


def test_iterate():
    test_data = [*list(range(10))]
    dl = DefaultDataloader(data=test_data)
    for i in range(20):
        try:
            val = next(dl)
        except StopIteration:
            val = None
        if i == 10:
            assert val == None, f"expected None but got {val}"
        elif i > 10:
            assert (i - 1) % 10 == val, f"expected {(i-1) % 10} but got {val}"
        elif i < 10:
            assert i % 10 == val, f"expected {i} but got {val}"
