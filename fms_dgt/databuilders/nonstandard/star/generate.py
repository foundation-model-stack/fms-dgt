# Standard
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import copy
import gc
import os
import shutil

# Third Party
from tqdm import trange

# Local
from fms_dgt.base.databuilder import DataBuilder, DataBuilderConfig
from fms_dgt.base.registry import get_data_builder, register_data_builder
from fms_dgt.base.task import DEFAULT_OUTPUT_DIR
from fms_dgt.blocks.generators.llm import MODEL_ID_OR_PATH
from fms_dgt.blocks.generators.vllm.vllm_serve import vLLMServerGenerator
from fms_dgt.blocks.trainers import BaseTrainerBlock
from fms_dgt.constants import BLOCKS_KEY, NAME_KEY, RUNNER_CONFIG_KEY, TASK_NAME_KEY
from fms_dgt.databuilders.nonstandard.star.task import StarTask
from fms_dgt.databuilders.transformation.cot.generate import CotTransformDataBuilder
from fms_dgt.datastores.default import DefaultDatastore
from fms_dgt.utils import init_dataclass_from_dict, sdg_logger

# all databuilders compatible with this framework
_VALID_TARGETS = [CotTransformDataBuilder]
_TGT_LLM_KEY = "llm1"


@dataclass
class StarDataBuilderConfig(DataBuilderConfig):
    """Class for STaR algorithm data builder"""

    target: Optional[Dict] = None


@register_data_builder("star_transform")
class StarTransformDataBuilder(DataBuilder):

    TASK_TYPE = StarTask

    # trainer
    trainer1: BaseTrainerBlock

    def __init__(
        self,
        *args: Any,
        config: Union[Mapping, StarDataBuilderConfig] = None,
        max_iters: int = 2,
        **kwargs: Any,
    ):
        # config first
        config: StarDataBuilderConfig = init_dataclass_from_dict(
            config, StarDataBuilderConfig
        )

        # init main class
        super().__init__(*args, config=config, **kwargs)

        assert (
            len(self._task_kwargs) == 1
        ), f"Cannot run STaR algorithm with more than one task at a time"

        self._meta_task: StarTask = self._tasks[0]

        # any other additional kwargs
        self._max_iters = max_iters

        self._kwargs = kwargs

    def execute_tasks(self):
        """Main entry point for task execution."""

        # first round
        target_databuilder, curr_iter_dir = self._init_iteration_data_builder(0)

        model_id_or_path = getattr(target_databuilder, _TGT_LLM_KEY).model_id_or_path
        assert os.path.exists(model_id_or_path), f"Must use a local model!"

        for iteration in trange(self._max_iters, desc="Bootstrap Iteration"):
            # execute databuilder
            target_databuilder.execute_tasks()

            # release model memory to allow for trainer
            target_databuilder.close()

            # train model
            model_id_or_path = self.trainer1(
                model_id_or_path=model_id_or_path,
                output_dir=curr_iter_dir,
                datastores=[
                    (
                        task.datastore,
                        self._meta_task.data_formatter_templates[task.name],
                    )
                    for task in target_databuilder.tasks
                ],
            )

            self.trainer1.release_model()

            # delete databuilder
            del target_databuilder
            gc.collect()

            # reload model with newly created
            if iteration != self._max_iters - 1:
                target_databuilder, curr_iter_dir = self._init_iteration_data_builder(
                    iteration + 1, model_id_or_path=model_id_or_path
                )

    def _init_iteration_data_builder(
        self, iteration: int, model_id_or_path: str = None
    ) -> Tuple[DataBuilder, str]:
        """Initializes data builder for this iteration"""

        meta_task_dir = os.path.join(DEFAULT_OUTPUT_DIR, self._meta_task.name)
        if (
            iteration == 0
            and self._meta_task.restart_generation
            and os.path.exists(meta_task_dir)
        ):
            shutil.rmtree(meta_task_dir)

        curr_iter_dir = os.path.join(meta_task_dir, f"iter_{iteration}")

        iter_task_kwargs = []
        for task_kwargs in self._meta_task.tasks:
            runner_config = copy.copy(task_kwargs[RUNNER_CONFIG_KEY])
            runner_config["output_dir"] = os.path.join(
                curr_iter_dir, task_kwargs.get(TASK_NAME_KEY)
            )
            task_kwargs = {**task_kwargs, RUNNER_CONFIG_KEY: runner_config}
            iter_task_kwargs.append(task_kwargs)

        config = copy.deepcopy(self.config.target)
        if model_id_or_path:
            for block in config[BLOCKS_KEY]:
                if block[NAME_KEY] == _TGT_LLM_KEY:
                    block[MODEL_ID_OR_PATH] = model_id_or_path

        # init target databuilder
        target_databuilder: DataBuilder = get_data_builder(
            self._meta_task.task_data_builder,
            **{
                "config": config,
                **self._kwargs,
                "task_kwargs": iter_task_kwargs,
            },
        )

        # validity checks
        if not type(target_databuilder) in _VALID_TARGETS:
            raise ValueError(
                f"Target databuilder is of type {type(target_databuilder)}, which is not one of valid targets {_VALID_TARGETS}"
            )
        if not hasattr(target_databuilder, _TGT_LLM_KEY):
            raise ValueError(
                f"Target databuilder must have an [{_TGT_LLM_KEY}] block to be used with STaR algorithm"
            )
        if not isinstance(
            getattr(target_databuilder, _TGT_LLM_KEY), vLLMServerGenerator
        ):
            raise ValueError(
                f"{_TGT_LLM_KEY} must be an instance of vLLMServerGenerator to be used with STaR algorithm"
            )
        for task in target_databuilder.tasks:
            if not isinstance(task.datastore, DefaultDatastore):
                raise ValueError(
                    f"Datastore for task [{task.name}] must be of type DefaultDatastore"
                )

        return target_databuilder, curr_iter_dir
