# Standard
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Mapping, Optional, Type, Union
import json
import os
import time

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.block import BaseBlock, get_row_name
from fms_dgt.base.registry import get_block
from fms_dgt.base.task import NAME_KEY, TYPE_KEY, SdgData, SdgTask
from fms_dgt.blocks.generators.llm import CachingLM
from fms_dgt.utils import all_annotations, sdg_logger


@dataclass
class DataBuilderConfig(dict):
    # data builder naming/registry
    name: Optional[str] = None
    blocks: Optional[dict] = None
    metadata: Optional[
        dict
    ] = None  # by default, not used in the code. allows for users to pass arbitrary info to data builders

    def __post_init__(self) -> None:
        pass


class DataBuilder(ABC):
    """A data builder represents a means of constructing data for a set of tasks"""

    VERSION: Optional[Union[int, str]] = None
    TASK_TYPE: SdgTask = SdgTask

    def __init__(
        self,
        config: Union[Mapping, DataBuilderConfig] = None,
        output_dir: str = None,
        max_gen_requests: int = None,
        task_inits: dict = None,
        task_kwargs: dict = None,
        **kwargs: Any,
    ) -> None:
        """ """
        if isinstance(config, DataBuilderConfig):
            self._config = config
        elif config is not None:
            self._config = DataBuilderConfig(**config)
        else:
            self._config = DataBuilderConfig()

        self._name = self.config.name
        self._max_gen_requests = max_gen_requests

        # initializing generators / validators
        self._init_blocks()

        # TODO: Data loader goes here
        self._tasks: List[SdgTask] = [
            self.TASK_TYPE(
                output_dir=output_dir, builder_cfg=config, **task_init, **task_kwargs
            )
            for task_init in task_inits
        ]
        #

        date_suffix = (
            datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )
        self._output_file_discarded = os.path.join(
            output_dir, f"discarded_{self.config.name}_{date_suffix}.log"
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def config(self) -> DataBuilderConfig:
        """Returns the DataBuilderConfig associated with this class."""
        return self._config

    @property
    def blocks(self) -> List[BaseBlock]:
        """Returns the blocks associated with this class."""
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

    def execute_tasks(self):
        # main entry point to task execution
        tasks = self._tasks + []

        # load the LM-generated data
        for task in tasks:
            task.load_data()
            if task.machine_data:
                sdg_logger.debug(
                    "Loaded %s machine-generated data", len(task.machine_data)
                )

        completed_tasks = [task for task in tasks if task.is_complete()]
        tasks = [task for task in tasks if task not in completed_tasks]

        # save task details for incomplete tasks
        for task in tasks:
            task.save_task()

        progress_bar = tqdm(total=len(tasks), desc="Running generation tasks")
        generate_start = time.time()

        request_idx = 0
        while tasks and request_idx <= self._max_gen_requests:
            request_idx += 1

            filtered_data: List[SdgData] = []
            for generated_inst in self.call_with_task_list(request_idx, tasks):
                # save incrementally
                task = next(
                    task for task in tasks if get_row_name(generated_inst) == task.name
                )
                task.save_data(generated_inst)
                filtered_data.append(generated_inst)

            for task in tasks:
                new_data = [
                    gen_inst
                    for gen_inst in filtered_data
                    if get_row_name(gen_inst) == task.name
                ]
                task.machine_data.extend(new_data)
                if task.is_complete():
                    completed_tasks.append(task)
                    progress_bar.update()

            tasks = [task for task in tasks if task not in completed_tasks]

            sdg_logger.info(
                "Generated %s data in this iteration, %s data overall",
                len(filtered_data),
                sum([len(task.machine_data) for task in tasks + completed_tasks]),
            )

        progress_bar.close()

        generate_duration = time.time() - generate_start
        sdg_logger.info("Generation took %.2fs", generate_duration)

    def call_with_task_list(
        self, request_idx: int, tasks: List[SdgTask]
    ) -> Iterable[SdgData]:
        # default behavior is to simply extract the seed / machine generated data and pass to data builder

        data_pool = [e for task in tasks for e in task.get_batch_examples()]
        args = [request_idx, data_pool]
        kwargs = dict()
        return self(*args, **kwargs)

    def __call__(
        self,
        request_idx: int,
        instruction_data: List[SdgData],
    ) -> List[SdgData]:
        """In this function we guarantee that no process outside of the user's control will make multiple parallel calls to this"""
        raise NotImplementedError


###
# Transformation-specific databuilder
###


class TransformationDataBuilder(DataBuilder):
    """
    This class is designed to have sensible default methods for transformation use cases
    """

    def execute_tasks(self):
        # main entry point to task execution
        tasks = self._tasks + []

        # load the LM-generated data
        for task in tasks:
            task.load_data()
            if task.machine_data:
                sdg_logger.debug(
                    "Loaded %s machine-generated data", len(task.machine_data)
                )

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
            task.save_data(generated_inst)
            filtered_data.append(generated_inst)

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

    def call_with_task_list(self, tasks: List[SdgTask]) -> Iterable[SdgData]:
        # default behavior is to simply extract the seed / machine generated data and pass to data builder
        data_pool = [e for task in tasks for e in task.get_batch_examples()]
        while data_pool:
            args = [data_pool]
            kwargs = dict()
            for output in self(*args, **kwargs):
                yield output
            data_pool = [e for task in tasks for e in task.get_batch_examples()]
