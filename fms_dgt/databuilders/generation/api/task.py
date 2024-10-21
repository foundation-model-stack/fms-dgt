# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import copy
import json
import random

# Local
from fms_dgt.base.task import SdgData, SdgTask, SdgTaskConfig


@dataclass
class ApiSdgTaskConfig(SdgTaskConfig):
    task_instruction: str = None
    api_specifications: Dict = None
    exclude_api_groups: List[str] = None
    min_func_count: int = 1
    max_func_count: int = 1
    check_arg_question_overlap: bool = True
    intent_only: bool = False
    single_function: bool = False
    require_nested: bool = False


@dataclass
class ApiSdgData(SdgData):
    """This class is intended to hold the seed / machine generated instruction data"""

    instruction: str
    input: str
    output: str
    positive_functions: List[str]
    seed_api_group: List[str]
    api_specifications: dict
    func_count_bounds: List[int]
    check_arg_question_overlap: bool
    intent_only: bool
    single_function: bool
    require_nested: bool

    def make_clear_copy(self):
        new_instr: ApiSdgData = copy.copy(self)
        (
            new_instr.input,
            new_instr.output,
            new_instr.positive_functions,
            new_instr.seed_api_group,
            new_instr.api_specifications,
        ) = (None, None, None, None, None)
        return new_instr

    def to_dict(self):
        output = asdict(self)
        output["api_specifications"] = None
        return output


class ApiSdgTask(SdgTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = ApiSdgData
    OUTPUT_DATA_TYPE = ApiSdgData
    CONFIG_TYPE = ApiSdgTaskConfig

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.all_api_specifications = {
            k: v
            for k, v in self.config.api_specifications.items()
            if (not self.config.exclude_api_groups)
            or k not in self.config.exclude_api_groups
        }
        self._api_specs_w_exclusions = self.config.api_specifications
        self._min_func_count = self.config.min_func_count
        self._max_func_count = self.config.max_func_count
        self._check_arg_question_overlap = self.config.check_arg_question_overlap
        self._intent_only = self.config.intent_only
        self._single_function = self.config.single_function
        self._require_nested = self.config.require_nested
        self._task_instruction = self.config.task_instruction

    @property
    def config(self) -> CONFIG_TYPE:
        return self._config

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            api_specifications=self._api_specs_w_exclusions[
                kwargs.get("seed_api_group")
            ],
            seed_api_group=kwargs.get("seed_api_group"),
            positive_functions=kwargs.get("positive_functions"),
            instruction=self._task_instruction,
            func_count_bounds=[self._min_func_count, self._max_func_count],
            check_arg_question_overlap=self._check_arg_question_overlap,
            intent_only=self._intent_only,
            single_function=self._single_function,
            require_nested=self._require_nested,
            input=kwargs.get("input"),
            output=kwargs.get("output"),
        )

    def instantiate_output_example(self, **kwargs: Any):
        kwargs.pop("api_specifications", None)
        return self.OUTPUT_DATA_TYPE(
            api_specifications=self._api_specs_w_exclusions[
                kwargs.get("seed_api_group")
            ],
            **kwargs,
        )

    def instantiate_instruction(self, data: ApiSdgData):
        keep_apis = (
            random.sample(
                list(data.api_specifications.keys()),
                k=min(len(data.api_specifications), 10),
            )
            + data.positive_functions
        )
        random.shuffle(keep_apis)
        data_copy = copy.copy(data)
        # data_copy.api_specifications = {
        #     k: data.api_specifications[k] for k in keep_apis
        # }
        data_copy.api_specifications = "\n".join(
            [json.dumps(data.api_specifications[k]) for k in keep_apis]
        )
        return super().instantiate_instruction(data_copy)
