# Standard
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

# Local
from fms_dgt.base.databuilder import DataBuilder, DataBuilderConfig
from fms_dgt.base.task import SdgTask
from fms_dgt.utils import sdg_logger


@dataclass
class PipelineConfig(DataBuilderConfig):
    data_schema: Dict = None


TYPE_KEY = "type"


class PipelineSdgTask(SdgTask):
    """This class is intended to hold general task information"""

    def __init__(self, data_schema: Dict, **kwargs):
        super().__init__(**kwargs)
        self._data_schema = data_schema

    @property
    def data_schema(self):
        return self._data_schema

    def get_example(self) -> Dict:
        try:
            example = next(self._dataloader)
            self._validate_example(example)
            return example
        except StopIteration:
            return None

    def get_batch_examples(self) -> List[Dict]:
        outputs = []
        for _ in range(self._dataloader_batch_size):
            example = self.get_example()
            if example is None:
                return outputs
            outputs.append(example)
        return outputs

    def _validate_example(self, ex: Dict):
        # validate example with schema
        return True

    def save_data(
        self,
        new_data: Union[Dict, List[Dict]],
    ) -> None:
        if type(new_data) != list:
            new_data = [new_data]
        self._datastore.save_data(new_data)

    def load_data(self) -> List[Dict]:
        loaded_data = self._datastore.load_data()
        if loaded_data:
            self.machine_data = loaded_data


class Pipeline(DataBuilder):
    """A data builder represents a means of constructing data for a set of tasks"""

    VERSION: Optional[Union[int, str]] = None
    TASK_TYPE: SdgTask = PipelineSdgTask

    def __init__(
        self,
        config: Mapping = None,
        task_kwargs: Dict = None,
        **kwargs: Any,
    ) -> None:
        """ """
        config: PipelineConfig = (
            PipelineConfig(**config) if config else PipelineConfig()
        )

        super().__init__(
            config=config,
            task_kwargs={"data_schema": config.data_schema, **task_kwargs},
            **kwargs,
        )

    def call_with_task_list(
        self, request_idx: int, tasks: List[PipelineSdgTask]
    ) -> Iterable[Dict]:
        _ = request_idx
        for task in tasks:
            sdg_logger.info(f"Running task: {task.name}")

            data_pool = task.get_batch_examples() + task.machine_data
            for res in self(data_pool):
                if "task_name" not in res:
                    res["task_name"] = task.name
                yield res

    def __call__(self, data_pool: List[Dict]):
        block_data = data_pool
        for block in self.blocks:
            sdg_logger.info(f"Running block {block.name}")
            block_data = block.generate(block_data)
        return block_data
