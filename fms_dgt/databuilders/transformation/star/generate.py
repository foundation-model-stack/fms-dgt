# Standard
from typing import Dict, List
import time

# Third Party
from tqdm import tqdm, trange

# Local
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.base.task import SdgData, SdgTask
from fms_dgt.base.trainer import Trainer
from fms_dgt.blocks.generators.llm import LMGenerator
from fms_dgt.databuilders.transformation.star.task import BootstrapTransformTask
from fms_dgt.utils import sdg_logger


@register_data_builder("star_transform")
class StarTransformDataBuilder(TransformationDataBuilder):

    TASK_TYPE: BootstrapTransformTask

    # llm1 is the main generator that will produce the synthetic examples
    llm1: LMGenerator

    # we are intentionally generic with val1 to maximize reuse
    val1: BaseValidatorBlock

    def __init__(
        self, task_kwargs: Dict, max_iters: int = 2, trainer_cfg: dict = None, **kwargs
    ):
        super().__init__(**kwargs)
        self._trainer_cfg = trainer_cfg
        self._task_kwargs = task_kwargs
        self._max_iters = max_iters

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
        """Main entry point for task execution."""

        # NOTE: here we are explicitly separating each task, i.e., we do not parallelize as we might in other databuilders
        for task_kwargs in tqdm(self._task_kwargs, desc="Running transformation tasks"):
            self.execute_single_task(task_kwargs)

        self.finalize_tasks(self._tasks)

    def execute_single_task(self, task_kwargs: BootstrapTransformTask):
        """Execute single task"""
        for iteration in trange(self._max_iters, desc="Bootstrap Iteration"):
            task = BootstrapTransformTask(iteration=iteration, **task_kwargs)
            task.save_task()

            # annotation of dataset
            self._annotate(task)

            trainer = Trainer(
                output_dir=task.curr_model_dir,
                datastore=task._datastore,
                **self._trainer_cfg,
            )
            trainer.train()

    def _annotate(self, task: BootstrapTransformTask):

        # resume from annotation
        task.load_dataloader_state()

        self.llm1.init_model(task.prev_model_dir)

        generate_start = time.time()

        new_data: List[SdgData] = []
        for generated_inst in self.call_with_task_list([task]):
            task.save_intermediate_data(generated_inst)
            new_data.append(generated_inst)
            task.save_dataloader_state()

        generate_duration = time.time() - generate_start
        sdg_logger.info(
            "Generation took %.2fs, generated %s data",
            generate_duration,
            len(task.machine_data),
        )

        self.llm1.release_model()

    def __call__(
        self,
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
