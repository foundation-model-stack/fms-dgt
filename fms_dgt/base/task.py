# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union
import abc
import json
import os

# Third Party
import pandas as pd

# Local
from fms_dgt.base.dataloader import DATA_PATH_KEY
from fms_dgt.base.registry import get_dataloader, get_datastore
from fms_dgt.dataloaders.default import DefaultDataloader
from fms_dgt.datastores.default import DefaultDatastore
from fms_dgt.utils import group_data_by_attribute

DEFAULT_OUTPUT_DIR = "output"


NAME_KEY = "name"
TYPE_KEY = "type"


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
        datastore: Optional[Dict] = None,
        restart_generation: Optional[bool] = False,
        builder_cfg: Optional[Mapping] = None,
        file_path: Optional[str] = None,
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

        ds_kwargs = {
            "task_name": name,
            "data_builder": data_builder,
            "restart_generation": restart_generation,
            "file_path": file_path,
            "builder_cfg": builder_cfg,
        }
        if datastore is None:
            self._datastore = DefaultDatastore(
                output_dir=output_dir, output_format=output_format, **ds_kwargs
            )
        else:
            assert (
                TYPE_KEY in datastore
            ), f"Must specify data store type with '{TYPE_KEY}' key"
            self._datastore = get_datastore(datastore.pop(TYPE_KEY))(
                **{**ds_kwargs, **datastore}
            )

        self._dataloader_batch_size = (
            dataloader_batch_size if dataloader_batch_size is not None else 10000000
        )
        dl_kwargs = {"seed_examples": seed_examples}
        if dataloader is None:
            self._dataloader = DefaultDataloader(data=seed_examples)
        else:
            assert TYPE_KEY in dataloader, (
                "Must specify dataloader type with %s key",
                TYPE_KEY,
            )
            self._dataloader = get_dataloader(dataloader.pop(TYPE_KEY))(
                **{**dl_kwargs, **dataloader}
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

    def save_data(
        self,
        new_data: Union[SdgData, List[SdgData]],
    ) -> None:
        if type(new_data) != list:
            new_data = [new_data]

        to_save = [d.to_output_dict() for d in new_data]
        self._datastore.save_data(to_save)

    def load_data(self) -> List[SdgData]:
        loaded_data = self._datastore.load_data()
        if loaded_data:
            self.machine_data = [
                self.instantiate_output_example(**d) for d in loaded_data
            ]

    def save_task(self):
        self._datastore.save_task()


T = TypeVar("T")


def group_data_by_task(data_list: List[T]) -> List[List[T]]:
    return group_data_by_attribute(data_list, "task_name")
