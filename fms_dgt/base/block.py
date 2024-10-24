# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import dataclasses

# Third Party
from datasets import Dataset
from ray.actor import ActorHandle
import pandas as pd
import ray

# Local
from fms_dgt.base.datastore import BaseDatastore, DatastoreDataType
from fms_dgt.base.registry import get_datastore
from fms_dgt.base.task_card import TaskRunCard
from fms_dgt.constants import DATASET_ROW_TYPE, DATASET_TYPE, TYPE_KEY


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


class BaseBlock(ABC):
    """Base Class for all Blocks"""

    def __init__(
        self,
        name: str = None,
        type: str = None,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        build_id: Optional[str] = None,
        builder_name: Optional[str] = None,
        datastore: Optional[Dict] = None,
        save_schema: Optional[List[str]] = None,
    ) -> None:
        """A block is a unit of computation that takes in some inputs and produces an output. It is intended to be specialized algorithms
            or processes that teams can contribute for others to use to build their pipelines.

        Args:
            name (str, optional): The name of the block.
            block_type (str, optional): The type of the block.

        Kwargs:
            arg_fields (Optional[List[str]], optional): A list of field names to use as positional arguments.
            kwarg_fields (Optional[List[str]], optional): A list of field names to use as keyword arguments.
            result_field (Optional[str], optional): Name of the result field in the input data row that the computation of the block will be written to.
            build_id (Optional[str], optional): ID to identify a particular SDG run.
            builder_name (Optional[str], optional): Name of the calling databuilder
            datastore (Optional[Dict]): A dictionary containing the configuration for the datastore.
            save_schema (Optional[List[str]], optional): The schema of the data that should be saved.

        Raises:
            TypeError: If any of the arguments are not of the correct type.
        """
        if not isinstance(arg_fields, (list, None.__class__)):
            raise TypeError("arg_fields must be of type 'list'")
        if not isinstance(kwarg_fields, (list, None.__class__)):
            raise TypeError("kwarg_fields must be of type 'list'")
        if not isinstance(result_field, (str, None.__class__)):
            raise TypeError("result_field must be of type 'str'")

        self._name = name
        self._block_type = type

        self._arg_fields = arg_fields
        self._kwarg_fields = kwarg_fields
        self._result_field = result_field

        self._save_schema = (
            save_schema
            or ((self._arg_fields or []) + (self._kwarg_fields or []))
            or None
        )
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
    def arg_fields(self) -> List[str]:
        """Returns a list of field names to use as positional arguments

        Returns:
            List[str]: A list of field names to use as positional arguments
        """
        return self._arg_fields

    @property
    def kwarg_fields(self) -> List[str]:
        """Returns a list of field names to use as keyword arguments

        Returns:
            List[str]: A list of field names to use as keyword arguments
        """
        return self._kwarg_fields

    @property
    def result_field(self) -> str:
        """Returns the name of the result field that computations will be written to

        Returns:
            str: Name of the result field that computations will be written to
        """
        return self._result_field

    @property
    def save_schema(self) -> List[str]:
        """Returns the schema of the data in the validator

        Returns:
            List[str]: Fields of the data
        """
        return self._save_schema

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

    def get_args_kwargs(
        self,
        inp: DATASET_ROW_TYPE,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Extracts the arguments and keyword arguments from a given input.

        Args:
            inp (DATASET_ROW_TYPE): The input data row.
            arg_fields (Optional[List[str]], optional): A list of field names to use as positional arguments.
            kwarg_fields (Optional[List[str]], optional): A list of field names to use as keyword arguments.

        Returns:
            Tuple[List[Any], Dict[str, Any]]: A tuple containing a list of positional arguments and a dictionary of keyword arguments.
        """
        arg_fields = arg_fields or self.arg_fields or []
        kwarg_fields = kwarg_fields or self.kwarg_fields or []

        if isinstance(inp, (dict, pd.DataFrame, Dataset)):
            return (
                [inp.get(arg) for arg in arg_fields],
                {kwarg: inp.get(kwarg) for kwarg in kwarg_fields},
            )
        raise TypeError(f"Unexpected input type: {type(inp)}")

    def write_result(
        self,
        inp: DATASET_ROW_TYPE,
        res: Any,
        result_field: Optional[str] = None,
    ) -> None:
        """Writes the result of the data processing step to the input data row.

        Args:
            inp (DATASET_ROW_TYPE): Input data row
            res (Any): Result to be written to the input data row
            result_field (Optional[str], optional): Name of the result field in the input data row.
        """
        result_field = result_field or self.result_field

        assert result_field is not None, "Result field cannot be None!"

        if isinstance(inp, (dict, pd.DataFrame, Dataset)):
            inp[result_field] = res
            return

        raise TypeError(f"Unexpected input type: {type(inp)}")

    def get_result(
        self,
        inp: DATASET_ROW_TYPE,
        result_field: Optional[str] = None,
    ) -> Any:
        """Gets the result that has been written onto object in `write_result` method

        Args:
            inp (DATASET_ROW_TYPE): Input is either a dict, pd.DataFrame, or Dataset
            result_field (Optional[str], optional): Field to access result from input object.

        Raises:
            TypeError: Raised if input type is not a dict, pd.DataFrame, or Dataset.

        Returns:
            Any: Object stored in result_field of input
        """
        result_field = result_field or self.result_field

        assert result_field is not None, "Result field cannot be None!"

        if isinstance(inp, (dict, pd.DataFrame, Dataset)):
            return inp[result_field]

        raise TypeError(f"Unexpected input type: {type(inp)}")

    def generate(self, *args, **kwargs):  # for interfacing with IL
        return self(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> DATASET_TYPE:
        """The generate function is the primary interface to a Block. Internally, it calls the `execute` method which contains the logic of the block."""
        return self.execute(*args, **kwargs)

    @abstractmethod
    def execute(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """The `execute` function is the primary logic of a Block

        Args:
            inputs (BLOCK_INPUT_TYPE): A block operates over a logical iterable
                of rows with named columns (see BLOCK_INPUT_TYPE)

        Kwargs:
            arg_fields (Optional[List[str]]): Names of fields within the rows of
                the inputs that should be extracted and passed as positional
                args to the underlying implementation methods.
            kwarg_fields (Optional[List[str]]): Names of fields within the rows
                of the inputs that should be extracted and passed as keyword
                args to the underlying implementation methods.
            **kwargs: Additional keyword args that may be passed to the derived
                block's generate function

        Returns:
            DATASET_TYPE: Input dataset with results added
        """
