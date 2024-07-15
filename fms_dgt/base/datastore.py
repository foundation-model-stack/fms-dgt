# Standard
from typing import List, TypeVar
import abc

DATA_PATH_KEY = "data_path"

T = TypeVar("T")


class BaseDatastore(abc.ABC):
    """Base Class for all dataloaders"""

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def save_data(self, new_data: List[T]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load_data(
        self,
    ) -> List[T]:
        raise NotImplementedError

    @abc.abstractmethod
    def save_task(
        self,
    ) -> None:
        raise NotImplementedError
