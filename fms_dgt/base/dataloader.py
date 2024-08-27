# Standard
from typing import Any
import abc


class BaseDataloader(abc.ABC):
    """Base Class for all dataloaders"""

    def __init__(self, **kwargs: Any) -> None:
        """Takes data from datastore object and produces one example to be used by SDG process."""
        super().__init__()

    def get_state(self) -> Any:
        """Gets the state of the dataloader, which influences __next__ function

        Returns:
            Any: Dataloader state
        """
        pass

    def set_state(self, state: Any) -> None:
        """Sets the state of the dataloader, which influences __next__ function

        Args:
            state (Any): Dataloader state
        """
        pass

    @abc.abstractmethod
    def __next__(self) -> Any:
        """Gets next element from dataloader

        Returns:
            Any: Element of dataloader
        """
        raise NotImplementedError
