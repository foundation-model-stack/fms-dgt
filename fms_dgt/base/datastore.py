# Standard
from enum import Enum
from typing import Any, Optional
import abc

# Local
from fms_dgt.base.block import DATASET_TYPE
from fms_dgt.base.task_card import TaskRunCard


class DatastoreDataType(Enum):
    MISC = 1
    TASK_DATA = 2
    CARD = 3
    VAL = 4
    STATE = 5
    SEED = 6
    FINAL_DATA = 7


class BaseDatastore(abc.ABC):
    """Base Class for all data stores"""

    def __init__(
        self,
        store_name: str,
        data_type: Optional[DatastoreDataType] = None,
        task_card: Optional[TaskRunCard] = None,
        restart: Optional[bool] = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._store_name = store_name
        self._data_type = data_type if data_type is not None else DatastoreDataType.MISC
        self._task_card = task_card
        self._restart = restart

    @property
    def store_name(self):
        return self._store_name

    @property
    def data_type(self):
        return self._data_type

    @property
    def task_card(self):
        return self._task_card

    def save_data(self, new_data: DATASET_TYPE) -> None:
        """
        Saves generated data to specified location

        Args:
            new_data (DATASET_TYPE): A list of data items to be saved
            task_card (Optional[TaskCard]): A task card corresponding to the data to be saved
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

    def close(self) -> None:
        """Method for closing a datastore when generation has completed"""
