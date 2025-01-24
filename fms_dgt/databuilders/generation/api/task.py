# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List
import copy
import json
import random

# Third Party
from openapi_schema_validator import OAS31Validator
import jsonschema

# Local
from fms_dgt.base.task import SdgData, SdgTask
from fms_dgt.utils import sdg_logger

_NAME = "name"


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
    allow_subset: bool

    def __post_init__(self):
        if self.positive_functions is None:
            try:
                f_call = json.loads(self.output)
                self.positive_functions = list(set([api[_NAME] for api in f_call]))
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not extract function names from provided annotation: {self.output}"
                )

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

    def __init__(
        self,
        *args: Any,
        task_instruction: str = None,
        api_specifications: Dict = None,
        exclude_api_groups: List[str] = None,
        min_func_count: int = 1,
        max_func_count: int = 1,
        check_arg_question_overlap: bool = True,
        intent_only: bool = False,
        single_function: bool = False,
        require_nested: bool = False,
        allow_subset: bool = False,
        **kwargs: Any,
    ):
        # TODO: Deprecate this
        api_specifications = _backwards_compatibility(api_specifications)

        super().__init__(*args, **kwargs)
        self.all_api_specifications = {
            k: v
            for k, v in api_specifications.items()
            if (not exclude_api_groups) or k not in exclude_api_groups
        }
        self._api_specs_w_exclusions = api_specifications
        self._min_func_count = min_func_count
        self._max_func_count = max_func_count
        self._check_arg_question_overlap = check_arg_question_overlap
        self._intent_only = intent_only
        self._single_function = single_function
        self._require_nested = require_nested
        self._task_instruction = task_instruction
        self._allow_subset = allow_subset

        _validate_data_schema(self.all_api_specifications.values())

    def instantiate_input_example(self, **kwargs: Dict):
        seed_api_group = kwargs.get(
            "seed_api_group",
            kwargs.get("domain", next(iter(self._api_specs_w_exclusions))),
        )
        return self.INPUT_DATA_TYPE(
            task_name=self.name,
            api_specifications=self._api_specs_w_exclusions[seed_api_group],
            seed_api_group=seed_api_group,
            positive_functions=kwargs.get("positive_functions"),
            instruction=self._task_instruction,
            func_count_bounds=[self._min_func_count, self._max_func_count],
            allow_subset=self._allow_subset,
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


def _backwards_compatibility(domains: Dict):
    def walk(d):
        if isinstance(d, dict):
            new_d = dict()
            for k, v in d.items():
                if k == "output_parameters":
                    new_d["outputs"] = v
                elif k == "parameters":
                    new_d = {**new_d, **v}
                else:
                    new_d[k] = walk(v)
            return new_d
        elif isinstance(d, list):
            return [walk(l) for l in d]
        else:
            return d

    for func_specs in domains.values():
        for func_info in func_specs.values():
            if func_info.get("parameters", dict()).get("properties"):
                return walk(domains)

    return domains


def _validate_data_schema(
    data_schemas_list: Iterable[Dict],
    warn_if_missing=["description", "properties"],
    error_if_missing=[],
):
    for data_schemas in data_schemas_list:
        for d_name, d_schema in data_schemas.items():
            for warn in warn_if_missing:
                if warn not in d_schema:
                    sdg_logger.warning(
                        "Expected to see field [%s] in data schema for [%s], instead got [%s]",
                        warn,
                        d_name,
                        d_schema,
                    )
            for err in error_if_missing:
                if err not in d_schema:
                    raise ValueError(
                        "Expected to see field [%s] in data schema for [%s], instead got [%s]",
                        err,
                        d_name,
                        d_schema,
                    )
            try:
                OAS31Validator.check_schema(d_schema)
            except jsonschema.exceptions.SchemaError as e:
                sdg_logger.error("Error detected in definition of [%s]", d_name)
                raise e
