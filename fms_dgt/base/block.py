# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# Third Party
from datasets import Dataset
import pandas as pd

# Local
from fms_dgt.base.task_card import TaskRunCard

DATASET_ROW_TYPE = Union[Dict[str, Any], pd.Series]
DATASET_TYPE = Union[Iterable[DATASET_ROW_TYPE], pd.DataFrame, Dataset]


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
        task_cards: Optional[List[TaskRunCard]] = None,
    ) -> None:
        """A block is a unit of computation that takes in some inputs and produces an output. It is intended to be specialized algorithms or processes that teams can contribute for others to use to build their pipelines.

        Args:
            name (str, optional): The name of the block.
            block_type (str, optional): The type of the block.
            arg_fields (Optional[List[str]], optional): A list of field names to use as positional arguments.
            kwarg_fields (Optional[List[str]], optional): A list of field names to use as keyword arguments.
            result_field (Optional[str], optional): Name of the result field in the input data row that the computation of the block will be written to.
            task_cards (Optional[TaskRunCard], optional): A list of all task cards this block is associated with.

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
        self._task_cards = task_cards

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

    @abstractmethod
    def generate(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        **kwargs,
    ) -> DATASET_TYPE:
        """The generate function is the primary interface to a Block

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
