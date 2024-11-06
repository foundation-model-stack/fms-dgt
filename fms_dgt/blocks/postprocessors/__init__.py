# Standard
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, List, Optional
import os
import shutil

# Third Party
import pyarrow as pa
import pyarrow.parquet as pq

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.constants import DATASET_TYPE, TASK_NAME_KEY


class BaseLargeScaleProcessingBlock(BaseBlock):
    """Base Class for all blocks that are intended to operate over files. The primary use for these is to
    more effecively interface with systems designed for high-volume data processing (e.g., DPK).
    """

    def __init__(
        self,
        processing_dir: str = None,
        data_path: Optional[str] = None,
        config_path: Optional[str] = None,
        restart: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """Processing block that accepts data, transforms it, and then reads the transformed data back to the databuilder

        Kwargs:
            processing_dir str: The path to the folder that will be used for processing data. Defaults to None.
            data_path (Optional[str]): A path from where data processing should begin
            config_path (Optional[str]): A path from where data processing should begin
            restart (Optional[bool]): Whether or not to restart processing from checkpoints if they exist
        """
        super().__init__(**kwargs)

        self._input_dir, self._logging_dir, self._output_dir = None, None, None

        self._config_path = config_path

        if processing_dir is None:
            raise ValueError("'processing_dir' is set to None for processing block")

        if os.path.exists(processing_dir) and restart:
            shutil.rmtree(processing_dir)

        self._input_dir = (
            os.path.join(processing_dir, "ds_proc_inputs")
            if data_path is None
            else data_path
        )
        self._intermediate_dir = os.path.join(processing_dir, "ds_proc_intermediate")
        self._logging_dir = os.path.join(processing_dir, "ds_proc_logging")
        self._output_dir = os.path.join(processing_dir, "ds_proc_outputs")

    @property
    def input_dir(self):
        return self._input_dir

    @property
    def intermediate_dir(self):
        return self._intermediate_dir

    @property
    def logging_dir(self):
        return self._logging_dir

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def config_path(self):
        return self._config_path

    def _set_data(
        self,
        inputs: DATASET_TYPE,
        file_map_fn: Callable,
        batch_size: Optional[int] = 10000,
    ) -> None:
        """Initializes the data directories for processing

        Args:
            inputs (DATASET_TYPE): Data to process
            file_map_fn (Callable): Mapping function that can be applied to input instance to determine what file to send it to

        Kwargs:
            batch_size (Optional[int]): batch size to read / write data
        """

        def write_batch(batch: List):
            batch_groups = defaultdict(list)
            for d in batch:
                batch_groups[file_map_fn(d)].append(d)
            for file_name, batch_group in batch_groups.items():
                records = pa.RecordBatch.from_pylist(batch_group)
                with pq.ParquetWriter(
                    os.path.join(self.input_dir, file_name + ".parquet"),
                    schema=records.schema,
                ) as writer:
                    writer.write_batch(records)

        os.makedirs(self.input_dir, exist_ok=True)

        batch = []
        for d in inputs:
            batch.append(d)
            if len(batch) > batch_size:
                write_batch(batch)
                batch.clear()
        if batch:
            write_batch(batch)

    def _get_data(self, batch_size: Optional[int] = 10000) -> DATASET_TYPE:
        """Gets data after processing has completed

        Kwargs:
            batch_size (Optional[int]): batch size to read / write data

        Returns:
            DATASET_TYPE: Resulting dataset
        """
        for f in os.listdir(self.output_dir):
            if f.endswith(".parquet"):
                parquet_file = pq.ParquetFile(os.path.join(self.output_dir, f))
                for batch in parquet_file.iter_batches(batch_size):
                    for d in batch.to_pylist():
                        yield d

    def execute(
        self,
        inputs: DATASET_TYPE,
        *args,
        file_map_fn: Optional[Callable] = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """Calls internal postprocessing block with _process method after storing data to parquet files

        Args:
            inputs (DATASET_TYPE): Dataset to process

        Kwargs:
            file_map_fn (Optional[Callable]): Mapping function that can be applied to input instance to determine what file to send it to

        Returns:
            DATASET_TYPE: Dataset after processing
        """

        if file_map_fn is None:
            file_map_fn = lambda x: x.get(TASK_NAME_KEY, "all_tasks")

        self._set_data(inputs, file_map_fn)

        self._process(*args, **kwargs)

        return self._get_data()

    @abstractmethod
    def _process(self, *args, **kwargs) -> None:
        """Method that executes the processing"""
