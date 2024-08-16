# Standard
from typing import Any, List, TypeVar
import abc

DATA_PATH_KEY = "data_path"

T = TypeVar("T")


class BaseDatastore(abc.ABC):
    """Base Class for all dataloaders"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    def save_data(self, new_data: List[T]) -> None:
        "Saves generated data to specified location"
        raise NotImplementedError

    def load_data(
        self,
    ) -> List[T]:
        "Loads generated data from save location"
        raise NotImplementedError

    def load_dataset(
        self,
    ) -> List[T]:
        "Loads dataset from specified source"
        raise NotImplementedError

    def save_task(
        self,
    ) -> None:
        "Default method for saving task specification"
        raise NotImplementedError

    def load_task(
        self,
    ) -> Any:
        "Default method for loading task specification"
        raise NotImplementedError

    def save_state(self, state: Any) -> None:
        "Saves a state object that can be used to restore an object (e.g., a dataloader) to a previous state"
        pass

    def load_state(
        self,
    ) -> Any:
        "Loads the state object"
        pass

    def save_instruction_data(self, new_data: List[T]) -> None:
        "Saves instruction data to specified location"
        pass

    def save_log_data(self, **kwargs):
        "Saves data regarding run information"
        pass
