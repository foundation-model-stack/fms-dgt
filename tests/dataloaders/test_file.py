# Standard
from typing import Dict, List
import copy
import os
import time

# Third Party
import pytest

# Local
from fms_dgt.dataloaders.file import FileDataloader


class TestFileDataloader:
    @pytest.mark.parametrize(
        "data_path", ["tests/dataloaders/test_data/test_seeds.yaml"]
    )
    def test_iterate(self, data_path):
        dl = FileDataloader(data_path=data_path)
        for i in range(10):
            try:
                val = next(dl)
            except StopIteration:
                val = None
            if i < 5:
                assert isinstance(val, dict)
                assert "question" in val
                assert "answer" in val
                assert f"{i+1}" in val["question"]
            if i == 5:
                assert val is None, f"expected None but got {val}"
