# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

# Local
from fms_dgt.base.task import SdgData, TransformTask
from fms_dgt.constants import TASK_NAME_KEY


@dataclass
class CotSdgData(SdgData):

    input: str
    output: str
    prompt: Optional[str] = None

    def to_dict(self):
        return {k: (v if k != "prompt" else None) for k, v in asdict(self).items()}


class CotTransformTask(TransformTask):

    INPUT_DATA_TYPE = CotSdgData

    def __init__(
        self,
        *args,
        prompt: str = None,
        data_formatter_template: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if prompt is None:
            raise ValueError(f"Cannot have [prompt] value as None")
        if data_formatter_template is None:
            raise ValueError(f"Cannot have [data_formatter_template] value as None")

        self._prompt = prompt
        self._data_formatter_template = data_formatter_template

    def instantiate_input_example(self, **kwargs: Any) -> CotSdgData:
        output = dict(self._data_formatter_template)
        for k in output.keys():
            for ds_k, ds_v in kwargs.items():
                inp_key = "{{" + ds_k + "}}"
                if inp_key in output[k]:
                    output[k] = output[k].replace(inp_key, str(ds_v))
        return CotSdgData(
            task_name=kwargs.get(TASK_NAME_KEY, self.name),
            prompt=self._prompt,
            **output,
        )

    def instantiate_instruction(self, data: CotSdgData) -> Dict:
        return data.to_dict()
