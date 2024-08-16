# Standard
from typing import Any
import abc


class BaseDataloader(abc.ABC):
    """Base Class for all dataloaders"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    def get_state(self) -> Any:
        pass

    def set_state(self, state: Any) -> None:
        pass

    @abc.abstractmethod
    def __next__(self) -> Any:
        raise NotImplementedError
