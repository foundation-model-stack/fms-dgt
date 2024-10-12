# Standard
from dataclasses import dataclass
from typing import Any, Dict
import os
import shutil

# Local
from fms_dgt.base.task import SdgData, TransformTask
from fms_dgt.constants import TASK_NAME_KEY
from fms_dgt.datastores.default import DefaultDatastore


@dataclass
class StarSdgData(SdgData):

    input: str
    output: str


class StarTransformTask(TransformTask):

    INPUT_DATA_TYPE = StarSdgData
    OUTPUT_DATA_TYPE = StarSdgData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            type(self._datastore) == DefaultDatastore
        ), f"Datastore must be of type {DefaultDatastore.__name__}"

        if self.restart_generation and os.path.exists(self._output_dir):
            shutil.rmtree(self._output_dir)

    def instantiate_input_example(self, **kwargs: Any) -> StarSdgData:
        return StarSdgData(
            task_name=kwargs.get(TASK_NAME_KEY, self.name),
            input=kwargs.get("input", kwargs.get("question")),
            output=kwargs.get("output", kwargs.get("answer")),
        )

    def instantiate_instruction(self, data: StarSdgData) -> Dict:
        return data.to_dict()
