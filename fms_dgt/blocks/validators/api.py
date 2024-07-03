# Standard
from typing import Any, Dict, List, Optional, Union
import json

# Third Party
from datasets import Dataset
from pandas import DataFrame

# First Party
from fms_dgt.base.block import BaseValidatorBlock
from fms_dgt.base.registry import register_block

# Constants

_NAME = "name"
_PARAM = "parameters"
_PROPERTIES = "properties"
_ARGS = "arguments"
_REQUIRED = "required"
_OUTPUT_PARAM = "output_parameters"

# Classes


@register_block("api_function_checking")
class APIGenSpecValidator(BaseValidatorBlock):
    """Class for API Sequence Prediction Validator"""

    def __call__(
        self,
        inputs: Union[List[Dict], DataFrame, Dataset],
        *args: Any,
        arg_fields: Optional[List[str]] = None,
        kwarg_fields: Optional[List[str]] = None,
        result_field: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        return super().__call__(
            inputs,
            *args,
            arg_fields=arg_fields,
            kwarg_fields=kwarg_fields,
            result_field=result_field,
            **kwargs,
        )

    def _validate(
        self,
        api_info: dict,
        question: str,
        answer: str,
        check_arg_question_overlap: bool = True,
        intent_only: bool = False,
        require_nested: bool = False,
        min_ct: int = 1,
        max_ct: int = 1,
    ) -> bool:

        try:
            sep_components = json.loads(answer)
        except json.decoder.JSONDecodeError as e:
            return False

        # check basic malformedness
        if type(sep_components) != list:
            return False

        # check for exact-match duplicates
        if len(set([str(x) for x in sep_components])) != len(sep_components):
            return False

        # check target count was hit
        if len(sep_components) < min_ct or len(sep_components) > max_ct:
            return False

        # ensure api names are covered
        component_names = set(
            [
                (component[_NAME] if _NAME in component else None)
                for component in sep_components
            ]
        )
        api_names = set([api[_NAME] for api in api_info.values()])

        if not (
            len(component_names.intersection(api_names))
            == len(component_names)
            == len(api_names)
        ):
            return False

        has_nested = False

        for i, component in enumerate(sep_components):

            # basic malformedness check
            if any([k for k in component.keys() if k not in [_NAME, _ARGS]]):
                return False

            if intent_only:

                # intent detection should not have arguments
                if _ARGS in component:
                    return False

                # we'll skip the rest of these checks if we're just interested in intent-detection
                continue

            # since we've checked that each component has an associated api, we can just grab the first (should be identical)
            matching_api = next(
                api for api in api_info.values() if api[_NAME] == component[_NAME]
            )
            matching_api_args = (
                matching_api[_PARAM][_PROPERTIES]
                if _PARAM in matching_api and _PROPERTIES in matching_api[_PARAM]
                else dict()
            )
            component_args = component[_ARGS] if _ARGS in component else dict()

            # filter if required args not met
            if (
                _PARAM in matching_api
                and _REQUIRED in matching_api[_PARAM]
                and set(matching_api[_PARAM][_REQUIRED]).difference(
                    component_args.keys()
                )
            ):
                return False

            # now do individual arg checking
            for arg_name, arg_content in component_args.items():

                # is argument name real
                if not arg_name in matching_api_args:
                    return False

                is_nested_call = require_nested and is_nested_match(
                    arg_content, sep_components[:i], api_info
                )
                has_nested = is_nested_call or has_nested

                # handle the case where slot values are not mentioned in the question
                if (
                    check_arg_question_overlap
                    and not is_nested_call
                    and str(arg_content).lower() not in question.lower()
                ):
                    return False

        return (require_nested and has_nested) or not (require_nested or has_nested)


def is_nested_match(arg_content: str, prev_components: List[Dict], api_info: Dict):
    for component in prev_components:
        matching_api = next(
            api for api in api_info.values() if api[_NAME] == component[_NAME]
        )
        if _OUTPUT_PARAM in matching_api:
            for out_param_name, out_param_info in matching_api[_OUTPUT_PARAM][
                _PROPERTIES
            ].items():
                nested_call = "$" + ".".join([component[_NAME], out_param_name])
                if nested_call == arg_content:
                    return True
    return False


@register_block("api_yes_no")
class ApiGenSpecYesNoValidation(APIGenSpecValidator):
    """Class for API Intent Detection Validator"""

    def _validate(self, api_info: dict, question: str, answer: str, **kwargs) -> bool:
        return answer in ["YES", "NO"]
