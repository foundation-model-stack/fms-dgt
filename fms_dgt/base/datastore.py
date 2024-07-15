# Standard
from typing import Any, List, TypeVar, Union
import abc

DATA_PATH_KEY = "data_path"

T = TypeVar("T")


class BaseDatastore(abc.ABC):
    """Base Class for all dataloaders"""

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def save(self, new_data: List[T]) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def load(
        self,
    ) -> List[T]:
        raise NotImplementedError
