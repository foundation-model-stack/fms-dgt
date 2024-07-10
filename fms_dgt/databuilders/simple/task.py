# Standard
from dataclasses import dataclass
from typing import Any, Optional

# Local
from fms_dgt.base.task import SdgData, SdgTask


@dataclass
class InstructLabSdgData(SdgData):
    """This class is intended to hold the seed / machine generated instruction data"""

    taxonomy_path: str
    task_description: str
    instruction: str
    input: str
    output: str
    document: str


class InstructLabSdgTask(SdgTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = InstructLabSdgData
    OUTPUT_DATA_TYPE = InstructLabSdgData

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            taxonomy_path=self.name,
            task_description=self.task_description,
            instruction=kwargs.get("question", kwargs.get("instruction")),
            input=kwargs.get("context", kwargs.get("input", "")),
            output=kwargs.get("answer", kwargs.get("output")),
            document=kwargs.get("document", None),
        )
