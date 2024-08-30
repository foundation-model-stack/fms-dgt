# Standard
from dataclasses import dataclass
import os

# Local
from fms_dgt.base.task import SdgData, TransformTask
from fms_dgt.datastores.default import DefaultDatastore


@dataclass
class BootstrapTransformData(SdgData):

    input: str
    output: str
    thought: str = None


class BootstrapTransformTask(TransformTask):

    INPUT_DATA_TYPE = BootstrapTransformData
    OUTPUT_DATA_TYPE = BootstrapTransformData

    def __init__(
        self,
        *args,
        iteration: int,
        output_dir: str = None,
        **kwargs,
    ):
        curr_output_dir = os.path.join(output_dir, f"iter_{iteration}")
        prev_output_dir = (
            os.path.join(output_dir, f"iter_{iteration-1}") if iteration else None
        )
        super().__init__(
            *args,
            output_dir=curr_output_dir,
            **kwargs,
        )
        assert (
            type(self._datastore) == DefaultDatastore
        ), f"Datastore must be of type {DefaultDatastore.__name__}"
        self._curr_model_dir = os.path.join(curr_output_dir, "models")
        self._prev_model_dir = (
            os.path.join(prev_output_dir, "models") if prev_output_dir else None
        )

    @property
    def curr_model_dir(self):
        return self._curr_model_dir

    @property
    def prev_model_dir(self):
        return self._prev_model_dir
