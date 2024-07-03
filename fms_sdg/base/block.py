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
