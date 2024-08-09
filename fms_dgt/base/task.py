# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, TypeVar, Union
import abc
import random

# Local
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
        seed_batch_size: Optional[int] = None,
        machine_batch_size: Optional[int] = None,
        seed_examples: Optional[List[Any]] = None,
        num_outputs_to_generate: Optional[int] = None,
    ):

        self._name = name
        self._task_description = task_description
        self._created_by = created_by
        self._data_builder = data_builder
        self._restart_generation = restart_generation
        self._file_path = file_path
        self._builder_cfg = builder_cfg
        self._seed_examples = seed_examples
        self._num_outputs_to_generate = num_outputs_to_generate
        self._output_format = output_format
        self._output_dir = output_dir

        # dataloader params
        self._dataloader_cfg = dataloader

        # datastore params
        self._datastore_cfg = datastore

        self.machine_data = []

        self._seed_batch_size = (
            seed_batch_size if seed_batch_size is not None else 10000000
        )
        if self._seed_batch_size < 0:
            raise ValueError(
                f"Cannot have negative value of {self._seed_batch_size} for seed_batch_size parameter"
            )

        self._machine_batch_size = (
            machine_batch_size if machine_batch_size is not None else 10000000
        )
        if self._machine_batch_size < 0:
            raise ValueError(
                f"Cannot have negative value of {self._machine_batch_size} for machine_batch_size parameter"
            )

        self.init_datastore()
        self.init_dataloader()

    def init_datastore(self):
        ds_kwargs = {
            "task_name": self._name,
            "data_builder": self._data_builder,
            "restart_generation": self._restart_generation,
            "file_path": self._file_path,
            "builder_cfg": self._builder_cfg,
            "seed_examples": self._seed_examples,
            "output_dir": self._output_dir,
            "output_format": self._output_format,
        }
        if self._datastore_cfg is None:
            self._datastore = DefaultDatastore(
                **ds_kwargs,
            )
        else:
            assert (
                TYPE_KEY in self._datastore_cfg
            ), f"Must specify data store type with '{TYPE_KEY}' key"
            self._datastore = get_datastore(
                self._datastore_cfg.get(TYPE_KEY),
                **{**ds_kwargs, **self._datastore_cfg},
            )

    def init_dataloader(self):
        if self._dataloader_cfg is None:
            self._dataloader = DefaultDataloader(datastore=self._datastore)
        else:
            assert TYPE_KEY in self._dataloader_cfg, (
                "Must specify dataloader type with %s key",
                TYPE_KEY,
            )
            self._dataloader = get_dataloader(
                self._dataloader_cfg.get(TYPE_KEY),
                datastore=self._datastore,
                **self._dataloader_cfg,
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

        # get outputs from seed data loader sequentially
        for _ in range(self._seed_batch_size):
            example = self.get_example()
            if example is None:
                break
            outputs.append(example)

        # get outputs from machine batch randomly
        m_data = self.machine_data
        if m_data and len(m_data) > self._machine_batch_size:
            m_data = random.sample(m_data, k=self._machine_batch_size)

        outputs.extend(m_data)

        return outputs

    def is_complete(self):
        return len(self.machine_data) > self.num_outputs_to_generate

    def save_data(
        self,
        new_data: Union[SdgData, List[SdgData]],
    ) -> None:
        if type(new_data) != list:
            new_data: List[SdgData] = [new_data]

        to_save = [d if type(d) == dict else d.to_output_dict() for d in new_data]
        self._datastore.save_data(to_save)

    def load_data(self) -> List[SdgData]:
        loaded_data = self._datastore.load_data()
        if loaded_data:
            self.machine_data = [
                self.instantiate_output_example(**d) for d in loaded_data
            ]

    def save_dataloader_state(self) -> None:
        self._datastore.save_state(self._dataloader.get_state())

    def load_dataloader_state(self) -> None:
        self._dataloader.set_state(self._datastore.load_state())

    def save_task(self):
        self._datastore.save_task()

    def load_task(self):
        return self._datastore.load_task()


###
# Transformation data classes
###


class TransformTask(SdgTask):
    def __init__(
        self, *args, seed_batch_size: int = 10, machine_batch_size: int = 0, **kwargs
    ):
        super().__init__(
            *args,
            seed_batch_size=seed_batch_size,
            machine_batch_size=machine_batch_size,
            **kwargs,
        )


T = TypeVar("T")


def group_data_by_task(data_list: List[T]) -> List[List[T]]:
    return group_data_by_attribute(data_list, "task_name")
