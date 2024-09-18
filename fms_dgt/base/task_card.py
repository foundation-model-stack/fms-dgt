# Standard
from dataclasses import asdict, dataclass
from typing import Optional
import uuid

_DEFAULT_EXEC_ID = "user"


@dataclass
class TaskRunCard:
    """This class is intended to hold the all information regarding the experiment being run"""

    task_name: str  # name of task
    databuilder_name: str  # name of databuilder associated with task
    task_spec: str  # json string capturing all of task settings
    databuilder_spec: str  # json string capturing all of databuilder settings
    exec_id: Optional[
        str
    ] = None  # id of entity executing the task (defaults to something generic)
    run_id: Optional[str] = None  # unique ID for the experiment

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = str(uuid.uuid4())
        if self.exec_id is None:
            self.exec_id = _DEFAULT_EXEC_ID

    def to_dict(self):
        return asdict(self)
