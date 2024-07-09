# Standard
from typing import Any
import abc

DATA_PATH_KEY = "data_path"


class BaseDataloader(abc.ABC):
    """Base Class for all dataloaders"""

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def __next__(self) -> Any:
        raise NotImplementedError
