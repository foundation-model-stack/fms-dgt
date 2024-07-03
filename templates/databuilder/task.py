# Standard
from dataclasses import dataclass
from typing import Any, Optional

# Local
from fms_dgt.base.task import SdgData, SdgTask


@dataclass
class TemplateSdgData(SdgData):
    """This class is intended to hold the seed / machine generated instruction data"""

    instruction: str
    input: str
    output: str


class TemplateSdgTask(SdgTask):
    """This class is intended to hold general task information"""

    DATA_TYPE = TemplateSdgData

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

    def instantiate_example(self, **kwargs: Any):
        """This is how one would read in an IL file, which has question / context / answer values for each example"""
        return self.DATA_TYPE(
            task_name=self.name,
            instruction=kwargs.get("question"),
            input=kwargs.get("context", ""),
            output=kwargs.get("answer"),
        )
