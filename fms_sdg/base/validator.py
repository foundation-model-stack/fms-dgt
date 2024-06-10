# Standard
from abc import ABC
from typing import Any, Dict, List, Tuple
import abc
import collections

# Local
from fms_sdg.base.generator import BaseGenerator
from fms_sdg.base.instance import Instance


class BaseValidator(ABC):
    """Base Class for all Validators"""

    def __init__(self, name: str, config: Dict) -> None:
        self._name = name
        self._config = config
        self._generators: List[BaseGenerator] = []
        self._validators: List[BaseValidator] = []

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

    @property
    def validators(self) -> List:
        """Returns the validators associated with this class."""
        return self._validators

    @abc.abstractmethod
    def validate_batch(self, inputs: List[Instance], **kwargs: Any) -> None:
        """Takes in a list of Instance objects (each containing their own arg / kwargs) and sets the results flag to true/false"""
        raise NotImplementedError
