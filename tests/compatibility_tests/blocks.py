# Standard
from typing import Any, Callable, Dict, List, Optional, Union
import json
import operator

# Third Party
from datasets import Dataset
from sdg.src.instructlab.sdg.filterblock import FilterByValueBlock
import pandas as pd

# First Party
from fms_dgt.base.block import BaseBlock


class TestFilterBlock(BaseBlock):
    """Base Class for all Blocks"""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._filter_block = FilterByValueBlock(
            filter_column=self.config.get("filter_column"),
            filter_value=self.config.get("filter_value"),
            operation=self.config.get("operation"),
        )

    def __call__(
        self,
        inputs: Union[List[Dict], pd.DataFrame, Dataset],
        **kwargs: Any,
    ) -> Any:
        return self._filter_block.generate(inputs)


def main():
    dataset = Dataset.from_dict(
        {"test": ["keep", "remove", "keep"], "nochange": ["data", "data", "data"]}
    )
    test_block = TestFilterBlock(
        name="block",
        config={
            "filter_column": "test",
            "filter_value": "remove",
            "operation": operator.ne,
        },
    )
    ret_dataset: Dataset = test_block(dataset)

    print(json.dumps(dataset.to_dict(), indent=4))
    print("\n=====\n")
    print(json.dumps(ret_dataset.to_dict(), indent=4))


if __name__ == "__main__":
    main()
