# Standard
from typing import Any, Dict, Optional

# Local
from fms_dgt.base.task import TransformTask
from fms_dgt.constants import TYPE_KEY


class PipelineTransformTask(TransformTask):
    """TransformTask is a subclass of SdgTask that has default values that are more conducive to transformation tasks."""

    def __init__(
        self,
        *args,
        dataloader: Optional[Dict] = None,
        seed_batch_size: int = 100000000,
        **kwargs,
    ):
        if dataloader is None:
            dataloader = {TYPE_KEY: "default", "loop_over_data": False}
        super().__init__(
            *args,
            dataloader=dataloader,
            seed_batch_size=seed_batch_size,
            **kwargs,
        )

    def instantiate_input_example(self, **kwargs: Any) -> Any:
        return kwargs

    def instantiate_output_example(self, **kwargs: Any) -> Any:
        return kwargs
