# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.databuilder import DataBuilderConfig, TransformationDataBuilder
from fms_dgt.base.registry import register_data_builder
from fms_dgt.blocks.compositions.sequence import BlockSequence
from fms_dgt.constants import DATASET_TYPE
from fms_dgt.databuilders.nonstandard.pipeline.task import PipelineTransformTask
from fms_dgt.utils import init_dataclass_from_dict, sdg_logger


@dataclass
class PipelineDataBuilderConfig(DataBuilderConfig):
    """Configuration for a data builder.

    Attributes:
        name (Optional[str]): The name of the data builder.
        blocks (Optional[List[Dict]]): A list of block configurations.
        postprocessors (Optional[List[str]]): A list of names of the blocks that should be used during postprocessing.
        metadata (Optional[Dict[str, Any]]): Metadata for the data builder. Allows for users to pass arbitrary info to data builders.
    """

    name: Optional[str] = None
    blocks: Optional[List[Union[Dict, BaseBlock]]] = None
    block_order: Optional[List[str]] = None
    block_params: Optional[List[Dict]] = None
    input_maps: Optional[List[Dict]] = None
    output_maps: Optional[List[Dict]] = None


@register_data_builder("transform_pipeline")
class PipelineTransformation(TransformationDataBuilder):
    """A pipeline is a config-based approach for constructing data for a set of tasks"""

    TASK_TYPE = PipelineTransformTask

    def __init__(
        self,
        config: Union[Mapping, PipelineDataBuilderConfig] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        config = init_dataclass_from_dict(config, PipelineDataBuilderConfig)
        super().__init__(config=config, *args, **kwargs)

    def _init_blocks(self):
        self._pipeline = BlockSequence(**asdict(self.config))
        self._blocks = self._pipeline.blocks

    def call_with_task_list(self, tasks: List[PipelineTransformTask]) -> Iterable[Dict]:
        for task in tasks:
            sdg_logger.info("Running task: %s", task.name)

            data_pool = task.get_batch_examples()

            for res in self(data_pool):
                res["task_name"] = task.name
                yield res

    def __call__(self, data_pool: DATASET_TYPE):
        return self._pipeline(data_pool)
