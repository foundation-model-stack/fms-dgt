# Standard
from typing import Any
import abc

# Local
from fms_dgt.base.block import DATASET_TYPE


class BaseDatastore(abc.ABC):
    """Base Class for all data stores"""

    def __init__(self, store_name: str, **kwargs: Any) -> None:
        super().__init__()
        self._store_name = store_name

    @property
    def store_name(self):
        return self._store_name

    def save_data(self, new_data: DATASET_TYPE) -> None:
        """
        Saves generated data to specified location

        Args:
            new_data (DATASET_TYPE): A list of data items to be saved
        """
        raise NotImplementedError

    def load_data(
        self,
    ) -> DATASET_TYPE:
        """Loads generated data from save location.

        Returns:
            A list of generated data of type DATASET_TYPE.
        """
        raise NotImplementedError
