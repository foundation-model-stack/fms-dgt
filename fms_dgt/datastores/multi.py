# Standard
from typing import Any, Dict, List, Optional, TypeVar

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import get_datastore, register_datastore
from fms_dgt.constants import TYPE_KEY
from fms_dgt.utils import sdg_logger

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

    def save_data(self, *args, **kwargs) -> None:
        """Saves generated data to specified location"""

        # save primary datastore first
        self._primary_datastore.save_data(*args, **kwargs)

        # try-catch for secondary
        for datastore in self._datastores[1:]:
            try:
                datastore.save_data(*args, **kwargs)
            except Exception as e:
                sdg_logger.debug(
                    f"Error encountered while uploading to secondary datastore:\n{str(e)}"
                )

    def load_data(
        self,
    ) -> List[T]:
        """Loads generated data from primary datastore"""
        return self._primary_datastore.load_data()

    def close(self) -> None:
        """Closes all datastores"""
        for datastore in self._datastores:
            datastore.close()
