# Standard
from typing import Any, Dict
import abc

DATA_PATH_KEY = "data_path"


class BaseDataloader(abc.ABC):
    """Base Class for all dataloaders"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    @abc.abstractmethod
    def __iter__(self) -> Any:
        raise NotImplementedError
