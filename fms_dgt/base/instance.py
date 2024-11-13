# Standard
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Instance:
    """This class is intended to hold the args / kwargs to a function call, which is useful when we want to organize a
    function's internal operations based on those kwargs"""

    args: Optional[tuple] = None
    kwargs: Optional[dict] = None
    result: Optional[Any] = None
    additional: Optional[dict[str, Any]] = None

    data: Optional[Any] = None
    idx: Optional[int] = None

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.kwargs is None:
            self.kwargs = dict()
