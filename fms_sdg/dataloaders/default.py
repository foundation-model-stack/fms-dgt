# Standard
from typing import Any, Callable, Dict, List

# Local
from fms_sdg.base.dataloader import BaseDataloader
from fms_sdg.base.registry import register_dataloader


@register_dataloader("default")
class DefaultDataloader(BaseDataloader):
    """Base Class for all dataloaders"""

    def __init__(
        self,
        *args: Any,
        data: List[Any] = None,
        proc_fn: Callable = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._data = data
        self._proc_fn = proc_fn

    def __iter__(self) -> Any:
        yield self._proc_fn(**self._data)
