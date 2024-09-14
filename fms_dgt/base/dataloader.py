# Standard
from typing import Any
import abc


class BaseDataloader(abc.ABC):
    """Base Class for all dataloaders"""

    def __init__(self, **kwargs: Any) -> None:
        """Takes data from datastore object and produces one example to be used by SDG process."""
        super().__init__()

    def save_state(self) -> None:
        """Saves the state of the dataloader which influences the __next__ function"""

    def load_state(self) -> None:
        """Loads the state of the dataloader and sets it which influences the __next__ function"""

    @abc.abstractmethod
    def __next__(self) -> Any:
        """Gets next element from dataloader

        Returns:
            Any: Element of dataloader
        """
        raise NotImplementedError
