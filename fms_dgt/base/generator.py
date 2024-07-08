# Standard
from abc import ABC
from typing import Any, Dict, List, Union
import abc


class BaseGenerator(ABC):
    """Base Class for all Generators"""

    def __init__(self, name: str, config: Dict, **kwargs: Any) -> None:
        self._name = name
        self._config: Dict = config
        self._generators: List[BaseGenerator] = []

        # overwrite config fields with kwargs (usually these will be command line args)
        self._config.update(kwargs)

    @property
    def name(self):
        return self._name

    @property
    def config(self):
        return self._config

    @property
    def generators(self) -> List:
        """Returns the generators associated with this class."""
        return self._generators

    @abc.abstractmethod
    def generate_batch(
        self, *args: Union[List, Dict], **kwargs: Union[str, Dict]
    ) -> None:
        raise NotImplementedError
