# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, TypeVar, Union
import abc
import json
import os

# Third Party
import pandas as pd

# Local
from fms_dgt.base.dataloader import DATALOADER_TYPE_KEY
from fms_dgt.base.registry import get_dataloader
from fms_dgt.dataloaders.default import DefaultDataloader
from fms_dgt.utils import group_data_by_attribute
import fms_dgt.dataloaders

DEFAULT_OUTPUT_DIR = "output"


@dataclass
class SdgData(abc.ABC):
    """This class is intended to hold the seed / machine generated instruction data"""

    task_name: str

    def to_output_dict(self):
        return asdict(self)


class SdgTask:
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = SdgData
    OUTPUT_DATA_TYPE = (
        INPUT_DATA_TYPE  # default output data type is the main type of the task
    )

    def __init__(
        self,
        name: str,
        task_description: str,
        created_by: str,
        data_builder: str,
        output_dir: Optional[str] = "output",
        output_format: Optional[str] = "jsonl",
        dataloader: Optional[Dict] = None,
        dataloader_batch_size: Optional[int] = None,
        seed_examples: Optional[List[Any]] = None,
        num_outputs_to_generate: Optional[int] = None,
    ):
        self._name = name
        self._task_description = task_description
        self._created_by = created_by
        self._data_builder = data_builder

        self._num_outputs_to_generate = num_outputs_to_generate
        self.machine_data = []

        self._output_dir = output_dir
        self._output_path = self._get_default_output_path(output_format)

        self._dataloader_batch_size = (
            dataloader_batch_size if dataloader_batch_size is not None else 10000000
        )

        if dataloader is None:
            self._dataloader = DefaultDataloader(data=seed_examples)
        else:
            assert (
                DATALOADER_TYPE_KEY in dataloader
            ), f"Must specify data loader type with '{DATALOADER_TYPE_KEY}' key"
            self._dataloader = get_dataloader(dataloader.pop(DATALOADER_TYPE_KEY))(
                **dataloader
            )

    @property
    def name(self):
        return self._name

    @property
    def task_description(self):
        return self._task_description

    @property
    def data_builder(self):
        return self._data_builder

    @property
    def output_path(self) -> str:
        return self._output_path

    @property
    def num_outputs_to_generate(self):
        return self._num_outputs_to_generate

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=kwargs.pop("task_name", self._name), **kwargs
        )

    def instantiate_output_example(self, **kwargs: Any):
        return self.OUTPUT_DATA_TYPE(**kwargs)

    def get_example(self) -> SdgData:
        try:
            return self.instantiate_input_example(**next(self._dataloader))
        except StopIteration:
            return None

    def get_batch_examples(self) -> List[SdgData]:
        outputs = []
        for _ in range(self._dataloader_batch_size):
            example = self.get_example()
            if example is None:
                return outputs
            outputs.append(example)
        return outputs

    def is_complete(self):
        return len(self.machine_data) > self.num_outputs_to_generate

    def _get_default_output_path(self, output_format: str = None):
        path_components = []
        path_components.append(self._output_dir)
        path_components.append(self._name)
        path_components.append("generated_instructions." + output_format)
        return os.path.join(*path_components)

    def save_data(
        self,
        new_data: Union[SdgData, List[SdgData]],
        output_path: str = None,
    ) -> None:
        if type(new_data) != list:
            new_data = [new_data]

        output_path = self._output_path if output_path is None else output_path
        output_format = os.path.splitext(output_path)[-1]

        if output_format == ".jsonl":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "a") as f:
                for d in new_data:
                    f.write(json.dumps(d.to_output_dict()) + "\n")
        elif output_format == ".parquet":
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame(new_data).to_parquet(
                output_path, engine="fastparquet", append=os.path.isfile(output_path)
            )
        else:
            raise ValueError(f"Unhandled output format: {output_format}")

    def load_data(self, output_path: str = None) -> List[SdgData]:
        output_path = self._output_path if output_path is None else output_path
        output_format = os.path.splitext(output_path)[-1]
        if output_format == ".jsonl":
            with open(output_path, "r") as f:
                try:
                    machine_data = [
                        self.instantiate_output_example(**json.loads(l.strip()))
                        for l in f.readlines()
                    ]
                except ValueError:
                    machine_data = []
        elif output_format == ".parquet":
            machine_data = [
                self.instantiate_output_example(**r)
                for r in (
                    pd.read_parquet(output_path, engine="fastparquet")
                    .apply(dict, axis=1)
                    .to_list()
                )
            ]
        else:
            raise ValueError(f"Unhandled output format: {output_format}")

        self.machine_data = machine_data

    def clear_data(self, output_path: str = None) -> List[SdgData]:
        output_path = self._output_path if output_path is None else output_path
        if os.path.exists(output_path):
            os.remove(output_path)


T = TypeVar("T")


def group_data_by_task(data_list: List[T]) -> List[List[T]]:
    return group_data_by_attribute(data_list, "task_name")
