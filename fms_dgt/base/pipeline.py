# Standard
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

# Local
from fms_dgt.base.block import DATASET_TYPE
from fms_dgt.base.databuilder import DataBuilder, DataBuilderConfig
from fms_dgt.base.task import SdgTask
from fms_dgt.blocks.compositions.chain import BlockChain
from fms_dgt.utils import sdg_logger


@dataclass
class PipelineConfig(DataBuilderConfig):
    data_map: Dict = None


class PipelineSdgTask(SdgTask):
    """This class is intended to hold general task information"""

    def __init__(self, data_map: Dict, **kwargs):
        super().__init__(**kwargs)
        self._data_map = data_map

    @property
    def data_map(self):
        return self._data_map

    def get_example(self) -> Dict:
        try:
            example = next(self._dataloader)
            example = self._map_example(example)
            return example
        except StopIteration:
            return None

    def _map_example(self, ex: Dict):
        # validate example with schema
        new_ex = dict()
        for k, v in ex.items():
            new_ex[self.data_map.get(k, k)] = v
        return new_ex

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
            task_kwargs={"data_map": config.data_map, **task_kwargs},
            **kwargs,
        )

    def _init_blocks(self):
        self._pipeline = BlockChain(self.config.blocks)

    def call_with_task_list(
        self, request_idx: int, tasks: List[PipelineSdgTask]
    ) -> Iterable[Dict]:
        _ = request_idx
        for task in tasks:
            sdg_logger.info(f"Running task: {task.name}")

            data_pool = task.get_batch_examples()

            for res in self(data_pool):
                if "task_name" not in res:
                    res["task_name"] = task.name
                yield res

    def __call__(self, data_pool: DATASET_TYPE):
        return self._pipeline.generate(data_pool)
