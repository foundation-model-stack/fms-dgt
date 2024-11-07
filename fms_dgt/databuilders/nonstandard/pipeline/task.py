# Standard
from typing import Any

# Local
from fms_dgt.base.task import TransformTask


class PipelineTransformTask(TransformTask):
    """TransformTask is a subclass of SdgTask that has default values that are more conducive to transformation tasks."""

    def __init__(
        self,
        *args,
        seed_batch_size: int = 100000000,
        **kwargs,
    ):
        super().__init__(*args, seed_batch_size=seed_batch_size, **kwargs)

    def instantiate_input_example(self, **kwargs: Any) -> Any:
        return kwargs

    def instantiate_output_example(self, **kwargs: Any) -> Any:
        return kwargs
