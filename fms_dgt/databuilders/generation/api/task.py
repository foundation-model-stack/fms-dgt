# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import copy

# Local
from fms_dgt.base.task import SdgData, SdgTask


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

    def to_output_dict(self):
        # we do this because api_specifications can be gigantic (e.g., Glaive)
        self_copy = copy.copy(self)
        self_copy.api_specifications = None
        dict_form = asdict(self_copy)
        dict_form.pop("api_specifications")
        return dict_form

    def make_clear_copy(self):
        new_instr: ApiSdgData = copy.copy(self)
        (
            new_instr.input,
            new_instr.output,
            new_instr.positive_functions,
            new_instr.seed_api_group,
        ) = (None, None, None, None)
        return new_instr


class ApiSdgTask(SdgTask):
    """This class is intended to hold general task information"""

    INPUT_DATA_TYPE = ApiSdgData
    OUTPUT_DATA_TYPE = ApiSdgData

    def __init__(
        self,
        task_instruction: str = None,
        api_specifications: Dict = None,
        exclude_api_groups: List[str] = None,
        min_func_count: int = 1,
        max_func_count: int = 1,
        check_arg_question_overlap: bool = True,
        intent_only: bool = False,
        single_function: bool = False,
        require_nested: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self._api_specifications = {
            k: v
            for k, v in api_specifications.items()
            if (not exclude_api_groups) or k not in exclude_api_groups
        }
        self._min_func_count = min_func_count
        self._max_func_count = max_func_count
        self._check_arg_question_overlap = check_arg_question_overlap
        self._intent_only = intent_only
        self._single_function = single_function
        self._require_nested = require_nested
        self._task_instruction = task_instruction

    def instantiate_input_example(self, **kwargs: Any):
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            api_specifications=self._api_specifications,
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
        return self.OUTPUT_DATA_TYPE(
            api_specifications=kwargs.pop(
                "api_specifications", self._api_specifications
            ),
            **kwargs,
        )
