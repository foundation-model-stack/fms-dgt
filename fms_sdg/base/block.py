# Standard
from abc import ABC
from typing import Any, Dict, List, Optional, Union
import abc

# Third Party
from datasets import Dataset
import pandas as pd


class BaseBlock(ABC):
    """Base Class for all Blocks"""

    def __init__(self, name: str, config: Dict, **kwargs: Any) -> None:
        self._name = name
        self._config: Dict = config
        self._blocks: List[BaseBlock] = []

        # overwrite config fields with kwargs (usually these will be command line args)
        self._config.update(kwargs)

        self._arg_fields = self._config.get("arg_fields", None)
        self._kwarg_fields = self._config.get("kwarg_fields", None)
        self._result_field = self._config.get("result_field", None)

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

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
        inputs: Union[List[Dict], pd.DataFrame, Dataset],
        *args: Any,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pass


class BaseGeneratorBlock(BaseBlock):
    pass


class BaseValidatorBlock(BaseBlock):
    def __init__(self, name: str, config: Dict, **kwargs: Any) -> None:
        super().__init__(name, config, **kwargs)
        self._filter_invalids = config.get("filter", False)

    def __call__(
        self,
        inputs: Union[List[Dict], pd.DataFrame, Dataset],
        *args: Any,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
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
