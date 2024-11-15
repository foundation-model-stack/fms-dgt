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
        save_schema: Optional[List[str]] = None,
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
            save_schema (Optional[List[str]], optional): The schema of the data that should be saved.

        Raises:
            TypeError: If any of the arguments are not of the correct type.
        """
        if not isinstance(input_map, (dict, list, None.__class__)):
            raise TypeError("[input_map] must be of type 'dict' or 'list'")
        if not isinstance(output_map, (dict, list, None.__class__)):
            raise TypeError("[output_map] must be of type 'dict' or 'list'")

        self._name = name
        self._block_type = type

        self._input_map = input_map
        self._output_map = output_map
        if self.DATA_TYPE is not None:
            self._req_args = [
                f.name
                for f in dataclasses.fields(self.DATA_TYPE)
                if f.default == dataclasses.MISSING and f.name != "SRC_DATA"
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
                    "schema": save_schema,
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

        inp_obj = asdict(inp) if is_dataclass(inp) else inp

        if isinstance(inp_obj, (dict, pd.DataFrame, Dataset)):

            # if nothing is provided, assume input matches
            if input_map is None and self.DATA_TYPE:
                input_map = {arg: arg for arg in self._req_args + self._opt_args}
            elif input_map is None and (
                self.DATA_TYPE is None or issubclass(self.DATA_TYPE, dict)
            ):
                # in this case, we assume the user wants to process their data as-is
                return inp_obj

            # NOTE: we flip this here because from a DGT pipeline, the input map goes from UserData -> BlockData
            data_type_map = {v: k for k, v in input_map.items()}

            mapped_data = {
                **{
                    r_a: inp_obj.get(data_type_map.get(r_a))
                    for r_a in self._req_args
                    if data_type_map.get(r_a) in inp_obj
                },
                **{
                    o_a: inp_obj.get(data_type_map.get(o_a))
                    for o_a in self._opt_args
                    if data_type_map.get(o_a) in inp_obj
                },
            }

            missing = [r_a for r_a in self._req_args if r_a not in mapped_data]
            if missing:
                raise ValueError(
                    f"Required inputs {missing} are not provided in 'input_map'"
                )

            return self.DATA_TYPE(**mapped_data, SRC_DATA=inp)

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

        if output_map is None and (
            self.DATA_TYPE is None or issubclass(self.DATA_TYPE, dict)
        ):
            # in this case, we assume the user wants to process their data as-is
            return inp

        # if none is provided, assume it maps to the input
        if output_map is None:
            output_map = {
                f.name: f.name
                for f in dataclasses.fields(self.DATA_TYPE)
                if f.name != "SRC_DATA"
            }

        if is_dataclass(inp.SRC_DATA):
            for k, v in output_map.items():
                # since a dataclass will throw an error, only try to add attributes if original data type has them
                if hasattr(inp.SRC_DATA, v):
                    setattr(inp.SRC_DATA, v, getattr(inp, k))
        elif isinstance(inp.SRC_DATA, (dict, pd.DataFrame, Dataset)):
            # TODO: handle things other than dictionaries
            for k, v in output_map.items():
                inp.SRC_DATA[v] = getattr(inp, k)
        else:
            raise TypeError(f"Unexpected input type: {type(inp)}")

        return inp.SRC_DATA

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
        """The generate function is the primary interface to a Block. Internally, it calls the `execute` method which contains the logic of the block.
            This function exists to have meta-processes (e.g., logging) that wrap around the core logic of a block

        Returns:
            DATASET_TYPE: Dataset resulting from processing contained in execute function
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
        *,
        fields: Optional[Union[List, Dict]] = None,
        result_field: Optional[str] = None,
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
