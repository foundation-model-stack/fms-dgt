# Standard
from abc import abstractmethod
from typing import Any, List, Optional, Tuple
import os
import shutil

# Third Party
import pyarrow as pa
import pyarrow.parquet as pq

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.datastore import BaseDatastore


class BasePostProcessingBlock(BaseBlock):
    """Base Class for all Postprocessors."""

    def __init__(
        self,
        processing_dir: str = None,
        data_path: Optional[str] = None,
        config_path: Optional[str] = None,
        restart: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """Post-processing block that accepts data, transforms it, and then reads the transformed data back to the databuilder

        Kwargs:
            processing_dir str: The path to the folder that will be used for processing data. Defaults to None.
            data_path (Optional[str]): A path from where data processing should begin
            config_path (Optional[str]): A path from where data processing should begin
            restart (Optional[bool]): Whether or not to restart processing from checkpoints if they exist
        """
        super().__init__(**kwargs)

        if processing_dir is None:
            raise ValueError(
                f"'processing_dir' cannot be none for post processing block"
            )

        if os.path.exists(processing_dir) and restart:
            shutil.rmtree(processing_dir)

        self._input_dir = (
            os.path.join(processing_dir, "post_proc_inputs")
            if data_path is None
            else data_path
        )
        self._intermediate_dir = os.path.join(processing_dir, "post_proc_intermediate")
        self._logging_dir = os.path.join(processing_dir, "post_proc_logging")
        self._output_dir = os.path.join(processing_dir, "post_proc_outputs")
        self._config_path = config_path

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
        file_name: str,
        from_datastore: BaseDatastore,
        batch_size: Optional[int] = 10000,
    ) -> None:
        """Initializes the data directories for post processing

        Args:
            file_name (str): filename to use for saving data
            from_datastore (BaseDatastore): data to write to input directory

        Kwargs:
            batch_size (Optional[int]): batch size to read / write data
        """

        def get_batches():
            batch = []
            for element in from_datastore.load_data():
                batch.append(element)
                if len(batch) >= batch_size:
                    yield pa.RecordBatch.from_pylist(batch)
                    batch = []
            if batch:
                yield pa.RecordBatch.from_pylist(batch)

        os.makedirs(self.input_dir, exist_ok=True)

        try:
            batches = get_batches()
            batch = next(batches)
        except StopIteration:
            return

        with pq.ParquetWriter(
            os.path.join(self.input_dir, file_name + ".parquet"), schema=batch.schema
        ) as writer:
            writer.write_batch(batch)
            for batch in batches:
                writer.write_batch(batch)

    def _save_data(
        self,
        file_name: str,
        to_datastore: BaseDatastore,
        batch_size: Optional[int] = 10000,
    ) -> None:
        """Saves data that after post processing has completed

        Args:
            file_name (str): filename used for original data
            to_datastore (BaseDatastore): datastore to write to data to

        Kwargs:
            batch_size (Optional[int]): batch size to read / write data
        """
        file_name = file_name + ".parquet"
        for f in os.listdir(self.output_dir):
            if f.endswith(file_name):
                parquet_file = pq.ParquetFile(os.path.join(self.output_dir, f))
                for batch in parquet_file.iter_batches(batch_size):
                    to_datastore.save_data(batch.to_pylist())

    def generate(
        self,
        datastores: List[Tuple[str, BaseDatastore, BaseDatastore]],
        *args,
        **kwargs,
    ) -> None:
        """Executes post processing on the data generated by a list of tasks

        Args:
            datastores (List[Tuple[str, BaseDatastore, BaseDatastore]]): A list containing tuples of the form <filename, from_datastore, to_datastore>,
                where filename is the name of the file that will be saved by the post processor, from_datastore is the datastore to load data from
                and to_datastore is the datastore to write data to
        """
        for filename, from_datastore, _ in datastores:
            self._set_data(filename, from_datastore)

        self._process(*args, **kwargs)

        for filename, _, to_datastore in datastores:
            self._save_data(filename, to_datastore)

    @abstractmethod
    def _process(self, *args, **kwargs) -> None:
        """Method that executes the post processing"""
