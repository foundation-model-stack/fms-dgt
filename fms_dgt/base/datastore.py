# Standard
from enum import Enum
from typing import Any, Optional
import abc

# Local
from fms_dgt.base.block import DATASET_TYPE


class DatastoreDataType(Enum):
    MISC = 1
    TASK_DATA = 2
    LOGGING = 3


class BaseDatastore(abc.ABC):
    """Base Class for all data stores"""

    def __init__(
        self,
        store_name: str,
        data_type: Optional[DatastoreDataType] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._store_name = store_name
        self._data_type = data_type if data_type is not None else DatastoreDataType.MISC

    @property
    def store_name(self):
        return self._store_name

    @property
    def data_type(self):
        return self._data_type

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

    def close(self):
        """Method for closing a datastore when generation has completed"""
