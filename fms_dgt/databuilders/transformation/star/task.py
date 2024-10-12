# Standard
from dataclasses import dataclass
from typing import Dict
import os

# Local
from fms_dgt.base.task import SdgData, TransformTask
from fms_dgt.datastores.default import DefaultDatastore


@dataclass
class StarSdgData(SdgData):

    input: str
    output: str


class StarTransformTask(TransformTask):

    INPUT_DATA_TYPE = StarSdgData
    OUTPUT_DATA_TYPE = StarSdgData

    def __init__(
        self,
        name: str,
        *args,
        iteration: int,
        output_dir: str = None,
        **kwargs,
    ):
        curr_output_dir = os.path.join(output_dir, name, f"iter_{iteration}")
        prev_output_dir = os.path.join(
            output_dir, name, f"iter_{(iteration - 1 if iteration else 'init')}"
        )
        self._curr_model_dir = os.path.join(curr_output_dir, "models")
        self._prev_model_dir = os.path.join(prev_output_dir, "models")

        super().__init__(
            name=name,
            *args,
            output_dir=curr_output_dir,
            **kwargs,
        )
        assert (
            type(self._datastore) == DefaultDatastore
        ), f"Datastore must be of type {DefaultDatastore.__name__}"

    @property
    def curr_model_dir(self):
        return self._curr_model_dir

    @property
    def curr_model(self):
        return os.path.join(self.curr_model_dir, "best")

    @property
    def prev_model_dir(self):
        return self._prev_model_dir

    @property
    def prev_model(self):
        return os.path.join(self.prev_model_dir, "best")

    @property
    def restart_generation(self):
        return self.restart_generation

    def instantiate_input_example(self, **kwargs: os.Any) -> StarSdgData:
        return StarSdgData(
            task_name=kwargs.get("task_name"),
            input=kwargs.get("input", kwargs.get("question")),
            output=kwargs.get("output", kwargs.get("answer")),
        )

    def instantiate_instruction(self, data: StarSdgData) -> Dict:
        return data.to_dict()
