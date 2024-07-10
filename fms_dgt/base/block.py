# Standard
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Union

# Third Party
from datasets import Dataset
import pandas as pd

BLOCK_ROW_TYPE = Union[Dict, pd.Series]
BLOCK_INPUT_TYPE = Union[Iterable[BLOCK_ROW_TYPE], pd.DataFrame, Dataset]


class BaseBlock(ABC):
    """Base Class for all Blocks"""

    def __init__(
        self,
        name: str = None,
        arg_fields: List[str] = None,
        kwarg_fields: List[str] = None,
        result_field: str = None,
    ) -> None:

        self._name = name

        # minor type checking
        if type(arg_fields) == str:
            arg_fields = [arg_fields]
        if type(kwarg_fields) == str:
            kwarg_fields = [kwarg_fields]
        if type(result_field) == list:
            assert (
                len(result_field) == 1
            ), "Cannot have multiple 'result' fields for {name}"
            result_field = result_field[0]

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

    @abstractmethod
    def generate(
        self,
        inputs: BLOCK_INPUT_TYPE,
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
        inputs: BLOCK_INPUT_TYPE,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
    ):
        outputs = []
        for x in inputs:
            inp_args, inp_kwargs = get_args_kwargs(
                x, arg_fields or self.arg_fields, kwarg_fields or self.kwarg_fields
            )
            res = self._validate(*inp_args, **inp_kwargs)
            if res or not self._filter_invalids:
                write_result(x, res, result_field or self.result_field)
                outputs.append(x)
        return outputs

    def _validate(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError


def get_args_kwargs(
    inp: BLOCK_ROW_TYPE,
    arg_fields: Optional[List[str]] = None,
    kwarg_fields: Optional[List[str]] = None,
):
    arg_fields = arg_fields or []
    kwarg_fields = or kwarg_fields or []

    if type(inp) == dict:
        args = [inp.get(arg) for arg in arg_fields]
        kwargs = {kwarg: inp.get(kwarg) for kwarg in kwarg_fields}
    elif type(inp) in [pd.DataFrame, Dataset]:
        args = [inp.get(arg) for arg in arg_fields]
        kwargs = {kwarg: inp.get(kwarg) for kwarg in kwarg_fields}
    else:
        raise ValueError(f"Unexpected input type: {type(inp)}")

    return args, kwargs


def write_result(
    inp: BLOCK_ROW_TYPE,
    res: Any,
    result_field: str,
):
    assert result_field is not None, "Result field cannot be None!"

    if isinstance(inp, (dict, pd.DataFrame, Dataset):
        inp[result_field] = res
    else:
        raise ValueError(f"Unexpected input type: {type(inp)}")
