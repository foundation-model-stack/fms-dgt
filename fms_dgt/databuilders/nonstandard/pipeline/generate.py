# Standard
from typing import Dict, Iterable, List

# Local
from fms_dgt.base.databuilder import TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.blocks.compositions.sequence import BlockSequence
from fms_dgt.constants import DATASET_TYPE
from fms_dgt.databuilders.nonstandard.pipeline.task import PipelineTransformTask
from fms_dgt.utils import sdg_logger


@register_data_builder("transform_pipeline")
class PipelineTransformation(TransformationDataBuilder):
    """A pipeline is a config-based approach for constructing data for a set of tasks"""

    TASK_TYPE = PipelineTransformTask

    def _init_blocks(self):
        self._pipeline = BlockSequence(self.config.blocks)
        self._blocks = self._pipeline.blocks

    def call_with_task_list(self, tasks: List[PipelineTransformTask]) -> Iterable[Dict]:
        for task in tasks:
            sdg_logger.info("Running task: %s", task.name)

            data_pool = task.get_batch_examples()

            for res in self(data_pool):
                if "task_name" not in res:
                    res["task_name"] = task.name
                yield res

    def __call__(self, data_pool: DATASET_TYPE):
        return self._pipeline(data_pool)
