# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Union

# Third Party
from datasets import Dataset
import pandas as pd

DATASET_ROW_TYPE = Union[Dict, pd.Series]
DATASET_TYPE = Union[Iterable[DATASET_ROW_TYPE], pd.DataFrame, Dataset]


class BaseBlock(ABC):
    """Base Class for all Blocks"""

    def __init__(
        self,
        name: str = None,
        arg_fields: List[str] = None,
        kwarg_fields: List[str] = None,
        result_field: str = None,
    ) -> None:

        if not (isinstance(arg_fields, list) or arg_fields is None):
            raise TypeError(f"arg_fields must be of type 'list'")
        if not (isinstance(kwarg_fields, list) or kwarg_fields is None):
            raise TypeError(f"kwarg_fields must be of type 'list'")
        if not (isinstance(result_field, str) or result_field is None):
            raise TypeError(f"result_field must be of type 'str'")

        self._name = name

        self._arg_fields = arg_fields
        self._kwarg_fields = kwarg_fields
        self._result_field = result_field

    @property
    def name(self):
        return self._name

    @property
    def arg_fields(self):
        return self._arg_fields

    @property
    def kwarg_fields(self):
        return self._kwarg_fields

    @property
    def result_field(self):
        return self._result_field

    def get_args_kwargs(
        self,
        inp: DATASET_ROW_TYPE,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
    ):

        arg_fields = arg_fields or self.arg_fields or []
        kwarg_fields = kwarg_fields or self.kwarg_fields or []

        if isinstance(inp, (dict, pd.DataFrame, Dataset)):
            args = [inp.get(arg) for arg in arg_fields]
            kwargs = {kwarg: inp.get(kwarg) for kwarg in kwarg_fields}
        else:
            raise TypeError(f"Unexpected input type: {type(inp)}")

        return args, kwargs

    def write_result(
        self,
        inp: DATASET_ROW_TYPE,
        res: Any,
        result_field: Optional[str] = None,
    ):
        result_field = result_field or self.result_field

        assert result_field is not None, "Result field cannot be None!"

        if isinstance(inp, (dict, pd.DataFrame, Dataset)):
            inp[result_field] = res
        else:
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
    ):
        """The generate function is the primary interface to a Block

        args:
            inputs (BLOCK_INPUT_TYPE): A block operates over a logical iterable
                of rows with named columns (see BLOCK_INPUT_TYPE)

        kwargs:
            arg_fields (Optional[List[str]]): Names of fields within the rows of
                the inputs that should be extracted and passed as positional
                args to the underlying implementation methods.
            kwarg_fields (Optional[List[str]]): Names of fields within the rows
                of the inputs that should be extracted and passed as keyword
                args to the underlying implementation methods.
            **kwargs: Additional keyword args that may be passed to the derived
                block's generate function
        """


class BaseUtilityBlock(BaseBlock):
    pass


class BaseGeneratorBlock(BaseBlock):
    pass


class BaseValidatorBlock(BaseBlock):
    def __init__(self, filter: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._filter_invalids = filter

    def generate(
        self,
        inputs: DATASET_TYPE,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
    ):
        outputs = []
        for x in inputs:
            inp_args, inp_kwargs = self.get_args_kwargs(x, arg_fields, kwarg_fields)
            res = self._validate(*inp_args, **inp_kwargs)
            if res or not self._filter_invalids:
                self.write_result(x, res, result_field)
                outputs.append(x)
        return outputs

    def _validate(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError
