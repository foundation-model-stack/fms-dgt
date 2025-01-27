# Standard
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List, Optional, Union
import dataclasses

# Third Party
from datasets import Dataset
import pandas as pd

# Local
from fms_dgt.base.datastore import BaseDatastore, DatastoreDataType
from fms_dgt.base.registry import get_datastore
from fms_dgt.base.task_card import TaskRunCard
from fms_dgt.constants import DATASET_ROW_TYPE, DATASET_TYPE, TYPE_KEY
from fms_dgt.utils import sdg_logger

_SRC_DATA = "SRC_DATA"


@dataclass
class BaseBlockData:
    """Internal data type for BaseBlock

    Attributes:
        SRC_DATA (Any): This attribute is used to store the original data. It SHOULD NOT be overwritten

    """

    SRC_DATA: Any

    def to_dict(self):
        return asdict(self)


class BaseBlock(ABC):
    """Base Class for all Blocks"""

    DATA_TYPE: BaseBlockData = None

    def __init__(
        self,
        name: str = None,
        type: str = None,
        input_map: Optional[Union[List, Dict]] = None,
        output_map: Optional[Union[List, Dict]] = None,
        build_id: Optional[str] = None,
        builder_name: Optional[str] = None,
        datastore: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """A block is a unit of computation that takes in some inputs and produces an output. It is intended to be specialized algorithms
            or processes that teams can contribute for others to use to build their pipelines.

        Args:
            name (str, optional): The name of the block.
            block_type (str, optional): The type of the block.

        Kwargs:
            input_map (Optional[Union[List, Dict]], optional): A mapping of field names from input objects to internal objects.
            output_map (Optional[Union[List, Dict]], optional): A mapping of field names from internal objects to output objects.
            build_id (Optional[str], optional): ID to identify a particular SDG run.
            builder_name (Optional[str], optional): Name of the calling databuilder
            datastore (Optional[Dict]): A dictionary containing the configuration for the datastore.

        Raises:
            TypeError: If any of the arguments are not of the correct type.
        """
        if not isinstance(input_map, (dict, list, None.__class__)):
            raise TypeError("[input_map] must be of type 'dict' or 'list'")
        if not isinstance(output_map, (dict, list, None.__class__)):
            raise TypeError("[output_map] must be of type 'dict' or 'list'")

        self._name = name
        self._block_type = type

        # input / output maps
        self._input_map = input_map
        self._output_map = output_map
        self._req_args, self._opt_args = [], []
        if not (self.DATA_TYPE is None or issubclass(self.DATA_TYPE, dict)):
            self._req_args = [
                f.name
                for f in dataclasses.fields(self.DATA_TYPE)
                if f.default == dataclasses.MISSING and f.name != _SRC_DATA
            ]
            self._opt_args = [
                f.name
                for f in dataclasses.fields(self.DATA_TYPE)
                if f.default != dataclasses.MISSING
            ]

        # datastore params
        self._datastore = None
        if datastore is not None:
            self._datastore = get_datastore(
                datastore.get(TYPE_KEY),
                **{
                    "task_card": TaskRunCard(
                        task_name=self.block_type,
                        databuilder_name=builder_name,
                        build_id=build_id,
                    ),
                    "data_type": DatastoreDataType.BLOCK,
                    "store_name": self.block_type,
                    **datastore,
                },
            )

        self._blocks: List[BaseBlock] = []

    @property
    def name(self) -> str:
        """Returns the name of the block

        Returns:
            str: The name of the block.
        """
        return self._name

    @property
    def block_type(self) -> str:
        """Returns a string representing type of the block

        Returns:
            str: The type of the block
        """
        return self._block_type

    @property
    def input_map(self) -> Union[List, Dict]:
        """Returns a dictionary or list that will be used to map field names from input objects to internal objects

        Returns:
            List[str]: A dictionary or list of fields to extract
        """
        return self._input_map

    @property
    def output_map(self) -> Union[List, Dict]:
        """Returns a dictionary or list that will be used to map field names from internal objects to output objects

        Returns:
            List[str]: A dictionary or list of fields to extract
        """
        return self._output_map

    @property
    def datastore(self) -> BaseDatastore:
        """Returns the datastore of the block

        Returns:
            BaseDatastore: Datastore of the block
        """
        return self._datastore

    def save_data(self, data: DATASET_TYPE) -> None:
        def to_serializable(x):
            def _to_serializable_inner(x):
                if isinstance(x, pd.Series):
                    return _to_serializable_inner(x.to_dict())
                elif dataclasses.is_dataclass(x):
                    return _to_serializable_inner(dataclasses.asdict(x))
                elif isinstance(x, dict):
                    return {k: _to_serializable_inner(v) for k, v in x.items()}
                elif isinstance(x, (tuple, list)):
                    return [_to_serializable_inner(y) for y in x]
                return x

            x = _to_serializable_inner(x)
            if not isinstance(x, dict):
                raise ValueError(
                    f"Attempting to serialize {x} to datastore, but data cannot be converted into dictionary"
                )
            return x

        if data and self._datastore is not None:
            self.datastore.save_data([to_serializable(x) for x in data])

    def transform_input(
        self,
        inp: Union[DATASET_ROW_TYPE, DATA_TYPE],  # type: ignore
        input_map: Dict,
    ) -> DATA_TYPE:  # type: ignore
        """Extracts the elements of the input as specified by map

        Args:
            inp (Union[DATASET_ROW_TYPE, DATA_TYPE]): The input data to be mapped
            input_map (Union[List, Dict]): A mapping of field names from input objects to internal objects.

        Returns:
            Dict: A dictionary containing the result of the mapping.
        """

        inp_obj = asdict(inp) if is_dataclass(inp) else dict(inp)

        # if none is provided, assume it maps to the src_data
        if input_map is None:
            input_map = self._get_default_map(inp)

        if isinstance(inp_obj, (dict, pd.DataFrame, Dataset)):
            # NOTE: we flip this here because from a DGT pipeline, the input map goes from UserData -> BlockData
            data_type_map = {v: k for k, v in input_map.items()}

            args = (self._req_args + self._opt_args) or data_type_map.keys()

            mapped_data = {
                arg: inp_obj.get(data_type_map.get(arg))
                for arg in args
                if data_type_map.get(arg) in inp_obj
            }

            missing = [r_a for r_a in self._req_args if r_a not in mapped_data]
            if missing:
                raise ValueError(
                    f"Required inputs {missing} are not provided in 'input_map'"
                )

            return (
                {**mapped_data, _SRC_DATA: inp}
                if self.DATA_TYPE is None
                else self.DATA_TYPE(**mapped_data, SRC_DATA=inp)
            )

        raise TypeError(f"Unexpected input type: {type(inp)}")

    def transform_output(
        self,
        inp: BaseBlockData,  # type: ignore
        output_map: Dict,
    ) -> Dict:
        """Extracts the elements of the internal data type as specified by output_map

        Args:
            inp (Union[DATASET_ROW_TYPE, DATA_TYPE]): The input data to be mapped
            output_map (Union[List, Dict]): A mapping of field names from input objects to internal objects.

        Returns:
            Dict: A dictionary containing the result of the mapping.
        """
        src_data = inp[_SRC_DATA] if isinstance(inp, dict) else inp.SRC_DATA

        # if none is provided, assume it maps to the src_data
        if output_map is None:
            output_map = self._get_default_map(src_data)

        if is_dataclass(src_data):
            for k, v in output_map.items():
                # since a dataclass will throw an error, only try to add attributes if original data type has them
                if hasattr(src_data, v):
                    attr_val = inp[k] if isinstance(inp, dict) else getattr(inp, k)
                    setattr(src_data, v, attr_val)
        elif isinstance(src_data, (dict, pd.DataFrame, Dataset)):
            # TODO: handle things other than dictionaries
            for k, v in output_map.items():
                attr_val = inp[k] if isinstance(inp, dict) else getattr(inp, k)
                src_data[v] = attr_val
        else:
            raise TypeError(f"Unexpected input type: {type(inp)}")

        return src_data

    def _get_default_map(self, data: Union[Dict, BaseBlockData]):
        # if DATA_TYPE is not provided, assume it maps to the input
        if is_dataclass(self.DATA_TYPE):
            fields = dataclasses.fields(self.DATA_TYPE)
        else:
            fields = data.keys() if isinstance(data, dict) else dataclasses.fields(data)
        fields = [f if isinstance(f, str) else f.name for f in fields]
        return {f: f for f in fields if f != _SRC_DATA}

    def generate(self, *args, **kwargs) -> DATASET_TYPE:  # for interfacing with IL
        """Method used to have compatibility with IL

        Returns:
            DATASET_TYPE: Resulting dataset output by __call__ function
        """
        return self(*args, **kwargs)

    def __call__(
        self,
        inputs: DATASET_TYPE,
        *args,
        input_map: Optional[Union[List, Dict]] = None,
        output_map: Optional[Union[List, Dict]] = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """The __call__ function is the primary interface to a Block. Internally, it calls the `execute` method which contains the logic of the block.
            This function exists to have meta-processes (e.g., logging) that wrap around the core logic of a block

        Args:
            inputs (DATASET_TYPE): Dataset to be processed by 'execute' method of block.
            input_map (Optional[Union[List, Dict]], optional): Mapping applied to each row of dataset that will convert row to instance of self.DATA_TYPE.
            output_map (Optional[Union[List, Dict]], optional): Mapping applied to each instance of self.DATA_TYPE that will convert instance back into row of dataset.

        Returns:
            DATASET_TYPE: Dataset resulting from processing contained in execute function.
        """
        input_map = input_map or self._input_map
        output_map = output_map or self._output_map

        transformed_inputs = map(lambda x: self.transform_input(x, input_map), inputs)
        if isinstance(inputs, (list, tuple)):
            transformed_inputs = type(inputs)(transformed_inputs)

        outputs = self.execute(transformed_inputs, *args, **kwargs)

        transformed_outputs = map(
            lambda x: self.transform_output(x, output_map), outputs
        )
        if isinstance(inputs, (list, tuple)):
            transformed_outputs = type(inputs)(transformed_outputs)

        return transformed_outputs

    @abstractmethod
    def execute(
        self,
        inputs: DATASET_TYPE,
        *args,
        input_map: Optional[Union[List, Dict]] = None,
        output_map: Optional[Union[List, Dict]] = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """The `execute` function is the primary logic of a Block

        Args:
            inputs (BLOCK_INPUT_TYPE): A block operates over a logical iterable
                of rows with named columns (see BLOCK_INPUT_TYPE)

        Kwargs:
            input_map (Optional[Union[List, Dict]], optional): A mapping of field names from input objects to internal objects.
            output_map (Optional[Union[List, Dict]], optional): A mapping of field names from internal objects to output objects.
            **kwargs: Additional keyword args that may be passed to the derived
                block's generate function

        Returns:
            DATASET_TYPE: Input dataset with results added
        """

    def close(self):
        """Method for safely deallocating all resources used by a block"""
        for block in self._blocks:
            block.close()


def get_row_name(gen_inst: DATASET_ROW_TYPE) -> str:
    """Gets the task name associated with the particular input instance.

    Args:
        gen_inst (DATASET_ROW_TYPE): The input to get the task name from.

    Returns:
        str: Name of task
    """
    if isinstance(gen_inst, dict):
        return gen_inst.get("task_name")
    else:
        return getattr(gen_inst, "task_name")
