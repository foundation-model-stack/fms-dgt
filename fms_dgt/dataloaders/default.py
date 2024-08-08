# Standard
from typing import Any

# Local
from fms_dgt.base.dataloader import BaseDataloader
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_dataloader


@register_dataloader("default")
class DefaultDataloader(BaseDataloader):
    """Base Class for all dataloaders"""

    def __init__(
        self,
        datastore: BaseDatastore = None,
    ) -> None:
        super().__init__()
        self._data = datastore.load_dataset()
        self._i = 0

    def get_state(self) -> int:
        return self._i

    def set_state(self, state: int) -> None:
        if state is not None:
            self._i = state

    def __next__(self) -> Any:
        try:
            value = self._data[self._i]
            self._i += 1
            return value
        except IndexError:
            # reset cycle
            self._i = 0
            raise StopIteration
