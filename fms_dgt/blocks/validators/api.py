# Standard
from typing import Dict, List
import json

# Third Party
from openapi_schema_validator import validate
import jsonschema

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.validators import BaseValidatorBlock

# Constants

_NAME = "name"
_PARAM = "parameters"
_PROPERTIES = "properties"
_ARGS = "arguments"
_LABEL = "label"
_OUTPUT_PARAM = "output_parameters"

# Classes


@register_block("api_function_checking")
class APIGenSpecValidator(BaseValidatorBlock):
    """Class for API Sequence Prediction Validator"""

    def _validate(
        self,
        api_info: dict,
        question: str,
        answer: str,
        check_arg_question_overlap: bool = True,
        intent_only: bool = False,
        require_nested: bool = False,
        allow_subset: bool = False,
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

        # ensure api names are covered
        component_names = set(
            [
                (component[_NAME] if _NAME in component else None)
                for component in sep_components
            ]
        )
        api_names = set([api[_NAME] for api in api_info.values()])

        # all api names must be present, but no additional api names
        if (allow_subset and not component_names.issubset(api_names)) or (
            not allow_subset and component_names.symmetric_difference(api_names)
        ):
            return False

        has_nested = False

        for i, component in enumerate(sep_components):

            # basic malformedness check
            if any(
                [
                    k
                    for k in component.keys()
                    if k not in ([_NAME, _ARGS] + ([_LABEL] if require_nested else []))
                ]
            ):
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

            # validate schema
            try:
                if _PARAM in matching_api:
                    validate(component_args, matching_api[_PARAM])
            except (
                jsonschema.exceptions.ValidationError,
                jsonschema.exceptions.SchemaError,
            ) as e:
                # if error is about a var label, e.g., $var1, then ignore error. Otherwise, raise error
                if not (require_nested and str(e).startswith("'$")):
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

        if _LABEL not in component:
            continue

        matching_api = next(
            api for api in api_info.values() if api[_NAME] == component[_NAME]
        )

        if _OUTPUT_PARAM in matching_api:
            for out_param_name, out_param_info in matching_api[_OUTPUT_PARAM][
                _PROPERTIES
            ].items():
                nested_call = ".".join([component[_LABEL], out_param_name])

                if nested_call == arg_content:
                    return True

    return False


@register_block("api_yes_no")
class ApiGenSpecYesNoValidation(APIGenSpecValidator):
    """Class for API Intent Detection Validator"""

    def _validate(self, api_info: dict, question: str, answer: str, **kwargs) -> bool:
        return answer in ["YES", "NO"]
