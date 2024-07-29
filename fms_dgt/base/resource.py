# Standard
from abc import ABC


class BaseResource(ABC):
    """Base Class for all shared Resources"""

    def __init__(self, id: str) -> None:
        self._id = id

    @property
    def id(self):
        return self._id

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other_resource: object):
        return self.id == other_resource.id
