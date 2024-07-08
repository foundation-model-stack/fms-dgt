# Standard
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Mapping, Optional, Union
import os
import time

# Third Party
from tqdm import tqdm

# Local
from fms_dgt.base.generator import BaseGenerator
from fms_dgt.base.registry import get_generator, get_validator
from fms_dgt.base.task import SdgData, SdgTask
from fms_dgt.base.validator import BaseValidator
from fms_dgt.generators.llm import CachingLM, LMGenerator
from fms_dgt.utils import all_annotations, sdg_logger


@dataclass
class DataBuilderConfig(dict):
    # data builder naming/registry
    name: Optional[str] = None
    generators: Optional[Union[str, list]] = None
    validators: Optional[Union[str, list]] = None
    generation_kwargs: Optional[dict] = None
    metadata: Optional[
        dict
    ] = None  # by default, not used in the code. allows for users to pass arbitrary info to data builders

    def __post_init__(self) -> None:
        if self.generation_kwargs is not None:
            if "temperature" in self.generation_kwargs:
                self.generation_kwargs["temperature"] = float(
                    self.generation_kwargs["temperature"]
                )


TYPE_KEY = "type"


class DataBuilder(ABC):
    """A data builder represents a means of constructing data for a set of tasks"""

    VERSION: Optional[Union[int, str]] = None
    TASK_TYPE: SdgTask = SdgTask

    def __init__(
        self,
        config: Mapping = None,
        lm_cache: str = None,
        output_dir: str = None,
        restart_generation: bool = False,
        max_gen_requests: int = None,
        task_inits: dict = None,
        task_kwargs: dict = None,
        **kwargs: Any,
    ) -> None:
        """ """
        self._config: DataBuilderConfig = (
            DataBuilderConfig(**config) if config else DataBuilderConfig()
        )
        self._name = self.config.name
        self._max_gen_requests = max_gen_requests
        self._restart_generation = restart_generation

        # initializing generators / validators
        self._init_gv(lm_cache=lm_cache)

        # TODO: Data loader goes here
        self._tasks: List[SdgTask] = [
            self.TASK_TYPE(output_dir=output_dir, **task_init, **task_kwargs)
            for task_init in task_inits
        ]
        #

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        date_suffix = (
            datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )
        self._output_file_discarded = os.path.join(
            output_dir, f"discarded_{self.config.name}_{date_suffix}.log"
        )

    @property
    def config(self) -> DataBuilderConfig:
        """Returns the DataBuilderConfig associated with this class."""
        return self._config

    @property
    def generators(self) -> List[BaseGenerator]:
        """Returns the generators associated with this class."""
        return self._generators

    @property
    def validators(self) -> List[BaseValidator]:
        """Returns the validators associated with this class."""
        return self._validators

    def _init_gv(self, lm_cache: str = None):
        _generators = (
            [self.config.generators]
            if type(self.config.generators) == str
            else self.config.generators
        )
        _validators = (
            [self.config.validators]
            if type(self.config.validators) == str
            else self.config.validators
        )
        self._generators: List[BaseGenerator] = []
        self._validators: List[BaseValidator] = []

        # TODO: need to handle nested generators / validators
        for i, info_src in enumerate([_generators, _validators]):
            # user may not define a generator / validator
            if info_src is not None:
                for obj_name, obj_config in info_src.items():
                    sdg_logger.debug(
                        "Initializing object %s with config %s", obj_name, obj_config
                    )
                    obj = (get_generator if i == 0 else get_validator)(
                        obj_config[TYPE_KEY]
                    )(obj_name, obj_config)

                    if lm_cache is not None and isinstance(obj, LMGenerator):
                        sdg_logger.info(
                            "Using cache at %s",
                            lm_cache + "_rank" + str(obj.rank) + ".db",
                        )
                        obj = CachingLM(
                            obj,
                            lm_cache
                            # each rank receives a different cache db.
                            # necessary to avoid multiple writes to cache at once
                            + f"_model{os.path.split(obj.model_id_or_path)[-1]}_rank{obj.rank}.db",
                        )

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
                    (self._generators if i == 0 else self._validators).append(obj)

    def execute_tasks(self):
        # main entry point to task execution
        tasks = self._tasks + []

        # load the LM-generated data
        for task in tasks:
            if self._restart_generation:
                task.clear_data()
            if os.path.exists(task.output_path):
                task.load_data()
                sdg_logger.debug(
                    "Loaded %s machine-generated data", len(task.machine_data)
                )

        completed_tasks = [task for task in tasks if task.is_complete()]
        tasks = [task for task in tasks if task not in completed_tasks]

        progress_bar = tqdm(total=len(tasks), desc="Running generation tasks")
        generate_start = time.time()

        request_idx = 0
        while tasks and request_idx <= self._max_gen_requests:
            request_idx += 1

            filtered_data: List[SdgData] = []
            for generated_inst in self.call_with_task_list(request_idx, tasks):
                # save incrementally
                task = next(
                    task for task in tasks if task.name == generated_inst.task_name
                )
                task.save_data(generated_inst)
                filtered_data.append(generated_inst)

            for task in tasks:
                new_data = [
                    gen_inst
                    for gen_inst in filtered_data
                    if gen_inst.task_name == task.name
                ]
                task.machine_data.extend(new_data)
                if task.is_complete():
                    completed_tasks.append(task)
                    progress_bar.update()

            tasks = [task for task in tasks if task not in completed_tasks]

            sdg_logger.info(
                "Generated %s data",
                sum([len(task.machine_data) for task in tasks + completed_tasks]),
            )

        progress_bar.close()

        generate_duration = time.time() - generate_start
        sdg_logger.info("Generation took %.2fs", generate_duration)

    def call_with_task_list(
        self, request_idx: int, tasks: List[SdgTask]
    ) -> Iterable[SdgData]:
        # default behavior is to simply extract the seed / machine generated data and pass to data builder
        data_pool = [e for task in tasks for e in (task.seed_data + task.machine_data)]
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
