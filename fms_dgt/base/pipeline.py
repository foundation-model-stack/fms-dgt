# Standard
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Union

# Local
from fms_dgt.base.databuilder import DataBuilder, DataBuilderConfig
from fms_dgt.base.task import SdgTask


@dataclass
class PipelineConfig(DataBuilderConfig):
    data_schema: Dict = None


TYPE_KEY = "type"


class Pipeline(DataBuilder):
    """A data builder represents a means of constructing data for a set of tasks"""

    VERSION: Optional[Union[int, str]] = None
    TASK_TYPE: SdgTask = SdgTask

    def __init__(
        self,
        config: Mapping = None,
        **kwargs: Any,
    ) -> None:
        """ """
        db_config = {
            f.name: config.get(f.name)
            for f in fields(DataBuilderConfig())
            if f.name in config
        }
        super().__init__(config=db_config, **kwargs)

        self._config: PipelineConfig = (
            PipelineConfig(**config) if config else PipelineConfig()
        )

        self._data_schema = self._config.data_schema
        print(self._data_schema)
        input("--")

    def generate(self, dataset):
        """
        Generate the dataset by running the pipeline steps.
        dataset: the input dataset
        """
        for block_prop in self.blocks:
            block_name = block_prop["name"]
            block_type = _lookup_block_type(block_prop["type"])
            block_config = block_prop["config"]
            drop_columns = block_prop.get("drop_columns", [])
            gen_kwargs = block_prop.get("gen_kwargs", {})
            drop_duplicates_cols = block_prop.get("drop_duplicates", False)
            block = block_type(self.ctx, self, block_name, **block_config)

            logger.info("Running block: %s", block_name)
            logger.info(dataset)

            dataset = block.generate(dataset, **gen_kwargs)

            # If at any point we end up with an empty data set, the pipeline has failed
            if len(dataset) == 0:
                raise EmptyDatasetError(
                    f"Pipeline stopped: Empty dataset after running block: {block_name}"
                )

            drop_columns_in_ds = [e for e in drop_columns if e in dataset.column_names]
            if drop_columns:
                dataset = dataset.remove_columns(drop_columns_in_ds)

            if drop_duplicates_cols:
                dataset = self._drop_duplicates(dataset, cols=drop_duplicates_cols)

        return dataset
