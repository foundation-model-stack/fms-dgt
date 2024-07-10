# Standard
from typing import Any, Callable, List

# Local
from fms_dgt.base.dataloader import BaseDataloader
from fms_dgt.base.registry import register_dataloader


@register_dataloader("default")
class DefaultDataloader(BaseDataloader):
    """Base Class for all dataloaders"""

    def __init__(
        self,
        data: List[Any] = None,
    ) -> None:
        super().__init__()
        self._data = data
        self._i = 0

    def __next__(self) -> Any:
        try:
            value = self._data[self._i]
            self._i += 1
            return value
        except IndexError:
            # reset cycle
            self._i = 0
            raise StopIteration
