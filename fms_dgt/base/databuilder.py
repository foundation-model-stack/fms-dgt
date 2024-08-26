# Standard
from abc import ABC
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Union
import json
import time

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.block import BaseBlock, get_row_name
from fms_dgt.base.registry import get_block
from fms_dgt.base.task import NAME_KEY, TYPE_KEY, SdgData, SdgTask, TransformTask
from fms_dgt.blocks.generators.llm import CachingLM
from fms_dgt.utils import all_annotations, sdg_logger


@dataclass
class DataBuilderConfig(dict):
    """Configuration for a data builder.

    Attributes:
        name (Optional[str]): The name of the data builder.
        blocks (Optional[List[Dict]]): A list of block configurations.
        metadata (Optional[Dict[str, Any]]): Metadata for the data builder. Allows for users to pass arbitrary info to data builders
    """

    name: Optional[str] = None
    blocks: Optional[dict] = None
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.blocks is None:
            self.blocks = []


class DataBuilder(ABC):
    """A data builder represents a means of constructing data for a set of tasks"""

    VERSION: Optional[Union[int, str]] = None
    TASK_TYPE: SdgTask = SdgTask

    def __init__(
        self,
        config: Union[Mapping, DataBuilderConfig] = None,
        max_gen_requests: int = None,
        max_stalled_requests: int = None,
        task_kwargs: dict = None,
        **kwargs: Any,
    ) -> None:
        """Initializes data builder object.

        Args:
            config (Union[Mapping, DataBuilderConfig], optional): Config specifying all databuilder settings.
            max_gen_requests (int, optional): Maximum number of data generation loop iterations to execute before terminating.
            max_stalled_requests (int, optional): Maximum number of data generation loop iterations that do not return new data before terminating.
            task_kwargs (List[dict], optional): List of task_kwargs for each task to be executed by this data builder.
        """

        if isinstance(config, DataBuilderConfig):
            self._config = config
        elif config is not None:
            self._config = DataBuilderConfig(**config)
        else:
            self._config = DataBuilderConfig()

        self._name = self.config.name

        self._max_gen_requests = (
            max_gen_requests if max_gen_requests is not None else float("inf")
        )
        self._max_stalled_requests = (
            max_stalled_requests if max_stalled_requests is not None else float("inf")
        )

        # initializing generators / validators
        self._init_blocks()

        # initialize tasks
        self._init_tasks(task_kwargs)

        self.kwargs = kwargs

    @property
    def name(self) -> str:
        """Returns the name of the data builder

        Returns:
            str: name string
        """
        return self._name

    @property
    def config(self) -> DataBuilderConfig:
        """Returns the DataBuilderConfig associated with this class.

        Returns:
            DataBuilderConfig: Config specifying data builder settings
        """
        return self._config

    @property
    def blocks(self) -> List[BaseBlock]:
        """Returns the blocks associated with this class.

        Returns:
            List[BaseBlock]: List of blocks to be used in this data builder
        """
        return self._blocks

    def _init_blocks(self):
        """This method does two things:

        (1) It initializes each block object specified in self.config.blocks
        (2) It sets the block-attributes for a DataBuilder to be those initialized blocks (where the block is assumed to be assigned to `obj_name`)
            - In the process of doing this, it checks that the type specified in the DataBuilder class's attribute matches the block type that was initialized

        This method is intended to be overloaded when type checking is not necessary (e.g., in the case of the Pipeline class).
        """
        self._blocks: List[BaseBlock] = []

        # TODO: need to handle nested blocks
        for obj_kwargs in self.config.blocks:

            for req_key in (NAME_KEY, TYPE_KEY):
                assert (
                    req_key in obj_kwargs
                ), f"'{req_key}' field missing in data builder config from block with args:\n{json.dumps(obj_kwargs, indent=4)} "

            obj_name = obj_kwargs.get("name")
            obj_type = obj_kwargs.get(TYPE_KEY)

            assert not any(
                block.name == obj_name for block in self._blocks
            ), f"Duplicate '{obj_name}' block in '{self.name}' data builder"

            obj = get_block(obj_type, **obj_kwargs)

            # we type check when not using a pipeline
            type_annotations = all_annotations(type(self))
            assert (
                obj_name in type_annotations
            ), f"Object {obj_name} is missing from definition of DataBuilder {self.__class__}"

            obj_type = type_annotations[obj_name]

            # double check types
            assert isinstance(obj, obj_type) or (
                isinstance(obj, CachingLM) and isinstance(obj.lm, obj_type)
            ), f"Type of retrieved object {obj.__class__} for {obj_name} does not match type {obj_type} specified in DataBuilder {self.__class__}"

            setattr(self, obj_name, obj)

    def _init_tasks(self, all_task_kwargs: List[dict]):
        """Initializes the tasks for this data builder

        Args:
            all_task_kwargs (List[dict]): List of task_kwargs for each task to be executed by this data builder
        """
        self._tasks: List[SdgTask] = [
            self.TASK_TYPE(builder_cfg=self._config, **task_kwargs)
            for task_kwargs in all_task_kwargs
        ]

    def execute_tasks(self):
        """Main entry point for task execution. Default behavior executes a loop until all tasks are complete, where each loop generates synthetic data."""

        # main entry point to task execution
        tasks = self._tasks + []

        # load the LM-generated data
        for task in tasks:
            task.load_intermediate_data()
            if task.machine_data:
                sdg_logger.debug(
                    "Loaded %s machine-generated data", len(task.machine_data)
                )
            task.load_dataloader_state()

        completed_tasks = [task for task in tasks if task.is_complete()]
        tasks = [task for task in tasks if task not in completed_tasks]

        # save task details for incomplete tasks
        for task in tasks:
            task.save_task()

        progress_bar = tqdm(total=len(tasks), desc="Running generation tasks")
        generate_start = time.time()

        stalled_cts = {task.name: self._max_stalled_requests for task in tasks}

        request_idx = 0
        while tasks and request_idx <= self._max_gen_requests:
            request_idx += 1

            filtered_data: List[SdgData] = []
            for generated_inst in self.call_with_task_list(request_idx, tasks):
                # save incrementally
                task = next(
                    task for task in tasks if get_row_name(generated_inst) == task.name
                )
                task.save_intermediate_data(generated_inst)
                filtered_data.append(generated_inst)
                task.save_dataloader_state()

            for task in tasks:
                new_data = [
                    gen_inst
                    for gen_inst in filtered_data
                    if get_row_name(gen_inst) == task.name
                ]
                task.machine_data.extend(new_data)

                stalled_cts[task.name] -= 1
                if new_data:
                    stalled_cts[task.name] = self._max_stalled_requests

                if task.is_complete() or stalled_cts[task.name] <= 0:
                    completed_tasks.append(task)
                    progress_bar.update()
                    if stalled_cts[task.name] <= 0:
                        sdg_logger.info(
                            "Task %s has not produced any data in the last %s attempts, terminating task",
                            task.name,
                            self._max_stalled_requests,
                        )

            tasks = [task for task in tasks if task not in completed_tasks]

            sdg_logger.info(
                "Generated %s data in this iteration, %s data overall",
                len(filtered_data),
                sum([len(task.machine_data) for task in tasks + completed_tasks]),
            )

        progress_bar.close()

        generate_duration = time.time() - generate_start
        sdg_logger.info("Generation took %.2fs", generate_duration)

        sdg_logger.info("Launch postprocessing")
        self.execute_postprocessing()
        sdg_logger.info("Postprocessing completed")

        self.finalize_tasks(completed_tasks)

    def call_with_task_list(
        self, request_idx: int, tasks: List[SdgTask]
    ) -> Iterable[SdgData]:
        """Executes data builder __call__ function for all in-progress tasks. Is executed in the inner loop of `execute_tasks`

        Args:
            request_idx (int): The iteration of `execute_tasks` this method was called at
            tasks (List[SdgTask]): List of in-progress tasks

        Returns:
            Iterable[SdgData]: List of data instances generated by the __call__ function
        """

        data_pool = [e for task in tasks for e in task.get_batch_examples()]
        args = [request_idx, data_pool]
        kwargs = dict()
        return self(*args, **kwargs)

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[SdgData],
    ) -> List[SdgData]:
        """Contains the main logic of a data builder. Takes in a list of data objects to be used as seed data and returns a list of data objects that reflect new instances

        Args:
            request_idx (int): The iteration of `execute_tasks` this method was called at
            instruction_data (List[SdgData]): List of data objects to be used as seed data

        Returns:
            List[SdgData]: List of new data objects that can be used for instruction-tuning
        """
        raise NotImplementedError

    def execute_postprocessing(self):
        """Executes any postprocessing required after tasks have completed."""
        pass

    def finalize_tasks(self, tasks: List[SdgTask]):
        """After tasks have completed, this method saves the final data for each task and saves any logging info.

        Args:
            tasks (List[SdgTask]): List of tasks that have completed
        """
        for task in tasks:
            task.save_final_data()
            task.save_log_data()


###
# Transformation-specific databuilder
###


class TransformationDataBuilder(DataBuilder):
    """This class is designed to have sensible default methods for transformation use cases"""

    def execute_tasks(self):
        """Main entry point for task execution. Default behavior iterates over all tasks and applies the transformation to each task's data."""
        tasks = self._tasks + []

        for task in tasks:
            assert isinstance(
                task, TransformTask
            ), f"Task {task.name} must inherit from TransformTask class to be used with TransformationDataBuilder"
            task.load_dataloader_state()

        # save task details for incomplete tasks
        for task in tasks:
            task.save_task()

        progress_bar = tqdm(total=len(tasks), desc="Running transformation tasks")
        generate_start = time.time()

        filtered_data: List[SdgData] = []
        for generated_inst in self.call_with_task_list(tasks):
            # save incrementally
            task = next(
                task for task in tasks if get_row_name(generated_inst) == task.name
            )
            task.save_intermediate_data(generated_inst)
            filtered_data.append(generated_inst)
            task.save_dataloader_state()

        for task in tasks:
            new_data = [
                gen_inst
                for gen_inst in filtered_data
                if get_row_name(generated_inst) == task.name
            ]
            task.machine_data.extend(new_data)
            progress_bar.update()

        sdg_logger.info(
            "Generated %s data",
            sum([len(task.machine_data) for task in tasks]),
        )

        progress_bar.close()

        generate_duration = time.time() - generate_start
        sdg_logger.info("Generation took %.2fs", generate_duration)

        self.finalize_tasks(tasks)

    def call_with_task_list(self, tasks: List[SdgTask]) -> Iterable[SdgData]:
        """Executes data builder __call__ function for all in-progress tasks.

        Args:
            tasks (List[SdgTask]): List of in-progress tasks

        Returns:
            Iterable[SdgData]: List of data instances generated by the __call__ function
        """
        # default behavior is to simply extract the seed / machine generated data and pass to data builder
        data_pool = [e for task in tasks for e in task.get_batch_examples()]
        while data_pool:
            args = [data_pool]
            kwargs = dict()
            for output in self(*args, **kwargs):
                yield output
            data_pool = [e for task in tasks for e in task.get_batch_examples()]
