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

    def to_output_dict(self) -> Dict:
        """Returns output dictionary representation of dataclass. Designed to be overridden with custom logic.

        Returns:
            Dict: Dictionary representation of dataclass
        """
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
        instruction_format: Optional[Dict[str, str]] = None,
        output_dir: Optional[str] = "output",
        output_format: Optional[str] = "jsonl",
        datastore: Optional[Dict] = None,
        restart_generation: Optional[bool] = False,
        builder_cfg: Optional[Mapping] = None,
        builder_dir: Optional[str] = None,
        file_path: Optional[str] = None,
        dataloader: Optional[Dict] = None,
        seed_batch_size: Optional[int] = None,
        machine_batch_size: Optional[int] = None,
        seed_examples: Optional[List[Any]] = None,
        num_outputs_to_generate: Optional[int] = None,
    ):
        """Initializes the Task object

        Args:
            name (str): The name of the Task object.
            task_description (str): A description of the SDG task is designed to solve.
            created_by (str): The name of the individual / group who created the code assistant.
            data_builder (str): The name of the data builder that should be used to process this task.
            instruction_format (Optional[Dict[str, str]]): A dictionary template that can be used to translate intermediate data objects to instruction-tuning pairs
            output_dir (Optional[str]): The directory where the generated outputs will be saved.
            output_format (Optional[str]): The format of the file where generated outputs are saved.
            datastore (Optional[Dict]): A dictionary containing the configuration for the datastore.
            restart_generation (Optional[bool]): A boolean indicating whether to restart generation from scratch.
            builder_cfg (Optional[Mapping]): A dictionary containing the configuration for the data builder.
            file_path (Optional[str]): The path to the task file.
            dataloader (Optional[Dict]): A dictionary containing the configuration for the dataloader.
            seed_batch_size (Optional[int]): The batch size used for seed examples.
            machine_batch_size (Optional[int]): The batch size used for machine examples.
            seed_examples (Optional[List[Any]]): A list of seed examples.
            num_outputs_to_generate (Optional[int]): The number of outputs to generate.
        """

        self._name = name
        self._task_description = task_description
        self._created_by = created_by
        self._data_builder = data_builder
        self._restart_generation = restart_generation
        self._file_path = file_path
        self._builder_cfg = builder_cfg
        self._builder_dir = builder_dir
        self._seed_examples = seed_examples
        self._num_outputs_to_generate = num_outputs_to_generate
        self._output_format = output_format
        self._output_dir = output_dir
        self._instruction_format = instruction_format

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

    def init_datastore(self) -> None:
        """Initialize datastore object for storing all SDG data."""

        ds_kwargs = {
            "task_name": self._name,
            "data_builder": self._data_builder,
            "restart_generation": self._restart_generation,
            "file_path": self._file_path,
            "builder_cfg": self._builder_cfg,
            "builder_dir": self._builder_dir,
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
        """Initialize the dataloader that passes all examples to SDG process."""

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
    def name(self) -> str:
        """Returns name of task.

        Returns:
            str: Name of task
        """
        return self._name

    @property
    def task_description(self) -> str:
        """Returns the task description.

        Returns:
            str: Task description
        """
        return self._task_description

    def instantiate_input_example(self, **kwargs: Any) -> INPUT_DATA_TYPE:
        """Instantiate an input example for this task. Designed to be overridden with custom initialization.

        Args:
            kwargs (Dict, optional): Kwargs used to instantiate an input example object.

        Returns:
            INPUT_DATA_TYPE: An instance of INPUT_DATA_TYPE.
        """
        return self.INPUT_DATA_TYPE(
            task_name=kwargs.pop("task_name", self._name), **kwargs
        )

    def instantiate_output_example(self, **kwargs: Any) -> OUTPUT_DATA_TYPE:
        """Instantiate an output example for this task. Designed to be overridden with custom initialization.

        Args:
            kwargs (Dict, optional): Kwargs used to instantiate an output example object.

        Returns:
            OUTPUT_DATA_TYPE: An instance of OUTPUT_DATA_TYPE.
        """
        return self.OUTPUT_DATA_TYPE(**kwargs)

    def instantiate_instruction(self, data: OUTPUT_DATA_TYPE) -> Dict:
        """Instantiates an instruction-tuning pair from output data instance.

        Args:
            data (OUTPUT_DATA_TYPE): Data to be converted to instruction-tuning pair.

        Returns:
            Dict: Dictionary representing an instruction-tuning pair.
        """
        data = asdict(data)
        output = dict(self._instruction_format)
        for k in output.keys():
            for ds_k, ds_v in data.items():
                inp_key = "{{" + ds_k + "}}"
                if inp_key in output[k]:
                    output[k] = output[k].replace(inp_key, str(ds_v))
        return output

    def get_example(self) -> SdgData:
        """Returns single example from dataloader.

        Returns:
            SdgData: Example to be used for SDG.
        """
        try:
            return self.instantiate_input_example(**next(self._dataloader))
        except StopIteration:
            return None

    def get_batch_examples(self) -> List[SdgData]:
        """Returns batch of examples from dataloader. Mixes examples from seed data and machine-generated data.

        Returns:
            List[SdgData]: List of examples to be used by SDG process.
        """
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

    def is_complete(self) -> bool:
        """Indicates whether SDG task has completed.

        Returns:
            bool: Whether task is complete or not.
        """
        return len(self.machine_data) > self._num_outputs_to_generate

    def save_intermediate_data(
        self,
        new_data: Union[SdgData, List[SdgData]],
    ) -> None:
        """Saves intermediate data produced during SDG (useful for checkpointing).

        Args:
            new_data (Union[SdgData, List[SdgData]]): List of SdgData to save.
        """
        if type(new_data) != list:
            new_data: List[SdgData] = [new_data]

        to_save = [d if type(d) == dict else d.to_output_dict() for d in new_data]
        self._datastore.save_data(to_save)

    def load_intermediate_data(self) -> List[SdgData]:
        """Loads intermediate data produced during SDG (will be used to resume SDG).

        Returns:
            List[SdgData]: List of SdgData that has been loaded
        """
        loaded_data = self._datastore.load_data()
        if loaded_data:
            self.machine_data = [
                self.instantiate_output_example(**d) for d in loaded_data
            ]

    def save_final_data(self) -> None:
        """Saves final instruction-tuning data that can be used directly for training."""
        if self._instruction_format is not None:
            loaded_data = self._datastore.load_data()
            if loaded_data:
                for d in loaded_data:
                    instruction = self.instantiate_instruction(
                        self.instantiate_output_example(**d)
                    )
                    self._datastore.save_instruction_data([instruction])

    def save_dataloader_state(self) -> None:
        """Saves state of data loader to enable resumption of SDG process later on."""
        self._datastore.save_state(self._dataloader.get_state())

    def load_dataloader_state(self) -> None:
        """Loads state of data loader to enable resumption of SDG process."""
        if not self._restart_generation:
            self._dataloader.set_state(self._datastore.load_state())

    def save_task(self) -> None:
        """Saves task specification to datastore."""
        self._datastore.save_task()

    def load_task(self) -> Any:
        """Loads task specification from datastore."""
        return self._datastore.load_task()

    def save_log_data(self) -> None:
        """Saves any Logging information to the datastore."""
        return self._datastore.save_log_data()


###
# Transformation data classes
###


class TransformTask(SdgTask):
    """TransformTask is a subclass of SdgTask that has default values that are more conducive to transformation tasks."""

    def __init__(
        self, *args, seed_batch_size: int = 10, machine_batch_size: int = 0, **kwargs
    ):
        super().__init__(
            *args,
            seed_batch_size=seed_batch_size,
            machine_batch_size=machine_batch_size,
            **kwargs,
        )

    def init_dataloader(self):
        """Initializes dataloader object for transform tasks."""
        if self._dataloader_cfg is None:
            self._dataloader = DefaultDataloader(
                datastore=self._datastore, loop_over_data=False
            )
        else:
            super().init_dataloader()


T = TypeVar("T")


def group_data_by_task(data_list: List[T]) -> List[List[T]]:
    """Utility function that groups input data by task name.

    Args:
        data_list (List[T]): List of SdgData to group into tasks

    Returns:
        List[List[T]]: SdgData that has been grouped into tasks
    """
    return group_data_by_attribute(data_list, "task_name")
