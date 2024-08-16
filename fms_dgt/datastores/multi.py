# Standard
from typing import Any, Dict, List, Optional, TypeVar

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import get_datastore, register_datastore
from fms_dgt.base.task import TYPE_KEY

T = TypeVar("T")


@register_datastore("multi_target")
class MultiTargetDatastore(BaseDatastore):
    """Class for all data stores"""

    def __init__(
        self,
        type: str,
        primary: Dict,
        additional: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> None:

        _ = type
        if additional is None:
            additional = []

        self._datastores: List[BaseDatastore] = []
        for datastore_cfg in [primary] + additional:
            assert (
                TYPE_KEY in datastore_cfg
            ), f"Must specify data store type with '{TYPE_KEY}' key"
            self._datastores.append(
                get_datastore(
                    datastore_cfg.get(TYPE_KEY),
                    **{**kwargs, **datastore_cfg},
                )
            )
        self._primary_datastore = self._datastores[0]

    @property
    def datastores(self):
        return self._datastores

    def save_data(self, new_data: List[T]) -> None:
        "Saves generated data to specified location"
        for datastore in self._datastores:
            datastore.save_data(new_data)

    def load_data(
        self,
    ) -> List[T]:
        "Loads generated data from primary datastore"
        return self._primary_datastore.load_data()

    def load_dataset(
        self,
    ) -> List[T]:
        "Loads dataset from specified source"
        return self._primary_datastore.load_dataset()

    def save_task(
        self,
    ) -> None:
        "Default method for saving task specification"
        for datastore in self._datastores:
            datastore.save_task()

    def load_task(
        self,
    ) -> Any:
        "Default method for loading task specification"
        return self._primary_datastore.load_task()

    def save_state(self, state: Any) -> None:
        "Saves a state object that can be used to restore an object (e.g., a dataloader) to a previous state"
        for datastore in self._datastores:
            datastore.save_state(state)

    def load_state(
        self,
    ) -> Any:
        "Loads the state object"
        return self._primary_datastore.load_state()

    def save_instruction_data(self, new_data: List[T]) -> None:
        "Saves instruction data to specified location"
        for datastore in self._datastores:
            datastore.save_instruction_data(new_data)

    def save_log_data(self, **kwargs):
        for datastore in self._datastores:
            datastore.save_log_data(**kwargs)
