# Standard
from dataclasses import asdict, dataclass
from typing import Optional
import uuid

_DEFAULT_BUILD_ID = "exp"


@dataclass
class TaskRunCard:
    """This class is intended to hold the all information regarding the experiment being run"""

    task_name: str  # name of task
    databuilder_name: str  # name of databuilder associated with task
    task_spec: Optional[str] = None  # json string capturing task settings
    databuilder_spec: Optional[str] = None  # json string capturing databuilder settings
    build_id: Optional[
        str
    ] = None  # id of entity executing the task (defaults to something generic)
    run_id: Optional[str] = None  # unique ID for the experiment

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = str(uuid.uuid4())
        if self.build_id is None:
            self.build_id = _DEFAULT_BUILD_ID

    def to_dict(self):
        return asdict(self)
