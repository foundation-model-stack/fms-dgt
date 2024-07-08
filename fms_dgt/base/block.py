# Standard
from abc import ABC
from typing import Any, Dict, List, Optional, Type, Union
import abc

# Third Party
from datasets import Dataset
import pandas as pd


class BaseBlock(ABC):
    """Base Class for all Blocks"""

    def __init__(
        self,
        name: str = None,
        arg_fields: List[str] = None,
        kwarg_fields: List[str] = None,
        result_field: str = None,
    ) -> None:

        assert name is not None, f"'name' field cannot be empty in block definition"

        self._name = name
        self._blocks: List[BaseBlock] = []

        self._arg_fields = arg_fields
        self._kwarg_fields = kwarg_fields
        self._result_field = result_field

    @property
    def name(self):
        return self._name

    @property
    def blocks(self) -> List:
        """Returns the constituent blocks associated with this class."""
        return self._blocks

    def get_args_kwargs(
        self,
        inp: Union[Dict, pd.DataFrame, Dataset],
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
    ):
        arg_fields = arg_fields if arg_fields is not None else self._arg_fields
        kwarg_fields = kwarg_fields if kwarg_fields is not None else self._kwarg_fields

        if arg_fields is None:
            arg_fields = []
        if kwarg_fields is None:
            kwarg_fields = []

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
        self, inp: Union[Dict, pd.DataFrame, Dataset], res: Any, result_field: str
    ):
        result_field = result_field if result_field is not None else self._result_field
        assert result_field is not None, "Result field cannot be None!"

        if type(inp) == dict:
            inp[result_field] = res
        elif type(inp) in [pd.DataFrame, Dataset]:
            inp[result_field] = res
        else:
            raise ValueError(f"Unexpected input type: {type(inp)}")

    @abc.abstractmethod
    def __call__(
        self,
        inputs: Union[List[Dict], Type[pd.DataFrame], Type[Dataset]],
        *args: Any,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        **kwargs: Any,
    ):
        pass


class BaseUtilityBlock(BaseBlock):
    pass


class BaseGeneratorBlock(BaseBlock):
    pass


class BaseValidatorBlock(BaseBlock):
    def __init__(self, filter: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._filter_invalids = filter

    def __call__(
        self,
        inputs: Union[List[Dict], Type[pd.DataFrame], Type[Dataset]],
        *args: Any,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
        **kwargs: Any,
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
