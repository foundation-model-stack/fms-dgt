# Standard
from abc import abstractmethod
from typing import Any, List, Optional
import os
import shutil

# Third Party
import pandas as pd

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock


class BasePostProcessingBlock(BaseBlock):
    """Base Class for all Postprocessors"""

    def __init__(
        self,
        processing_dir: str = None,
        data_path: Optional[str] = None,
        config_path: Optional[str] = None,
        restart: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        """Post-processing block that accepts data, transforms it, and then writes the transformed data to the original datastore

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
    def data_filename(self):
        return f"{self.block_type}_{self.name}.parquet"

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

    def _set_data(self, data: DATASET_TYPE):
        """Initializes the data directories for post processing"""
        os.makedirs(self.input_dir)
        data_path = os.path.join(self.input_dir, self.data_filename)
        pd.DataFrame(data).to_parquet(
            data_path,
            engine="fastparquet",
        )

    def _get_data(self, inputs: DATASET_TYPE) -> DATASET_TYPE:
        """Gets data that after post processing has completed

        Args:
            inputs (DATASET_TYPE): Original input data

        Returns:
            DATASET_TYPE: Output data that has been post processed
        """
        ret_data = []
        for f in os.listdir(self.output_dir):
            if f.endswith(self.data_filename):
                data_path = os.path.join(self.output_dir, f)
                proc_data = (
                    pd.read_parquet(data_path, engine="fastparquet")
                    .apply(dict, axis=1)
                    .to_list()
                )
                ret_data.extend(proc_data)
        # TODO: make this more efficient, e.g., stream
        if isinstance(inputs, pd.DataFrame):
            return pd.DataFrame(ret_data)
        else:
            return ret_data

    def generate(
        self,
        inputs: DATASET_TYPE,
        *args,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Executes post processing on the data generated by a list of tasks

        Args:
            inputs (BLOCK_INPUT_TYPE): A block operates over a logical iterable
                of rows with named columns (see BLOCK_INPUT_TYPE)

        Kwargs:
            arg_fields (Optional[List[str]], optional): A list of field names to use as positional arguments.
            kwarg_fields (Optional[List[str]], optional): A list of field names to use as keyword arguments.
            result_field (Optional[str], optional): Name of the result field in the input data row that the computation of the block will be written to.
        """
        self._set_data(inputs)
        self._process(
            *args,
            arg_fields=arg_fields,
            kwarg_fields=kwarg_fields,
            result_field=result_field,
            **kwargs,
        )
        return self._get_data(inputs)

    @abstractmethod
    def _process(self, *args, **kwargs) -> None:
        """Method that executes the post processing"""
