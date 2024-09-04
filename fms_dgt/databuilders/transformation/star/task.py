# Standard
from dataclasses import dataclass
import os

# Local
from fms_dgt.base.task import SdgData, TransformTask
from fms_dgt.datastores.default import DefaultDatastore


@dataclass
class BootstrapInputData(SdgData):

    question: str
    answer: str
    thought: str = None


@dataclass
class BootstrapOutputData(SdgData):

    input: str
    output: str
    thought: str = None


class BootstrapTransformTask(TransformTask):

    INPUT_DATA_TYPE = BootstrapInputData
    OUTPUT_DATA_TYPE = BootstrapOutputData

    def __init__(
        self,
        name: str,
        *args,
        iteration: int,
        output_dir: str = None,
        **kwargs,
    ):
        curr_output_dir = os.path.join(output_dir, name, f"iter_{iteration+1}")
        prev_output_dir = os.path.join(output_dir, name, f"iter_{iteration}")
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
