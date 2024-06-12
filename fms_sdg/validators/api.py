# Standard
from typing import Any, List
import json

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.base.registry import register_validator
from fms_sdg.base.validator import BaseValidator

# Constants

_NAME = "name"
_PARAM = "parameters"
_PROPERTIES = "properties"
_ARGS = "arguments"
_REQUIRED = "required"

# Classes


@register_validator("api_function_checking")
class APIGenSpecValidator(BaseValidator):
    """Class for API Sequence Prediction Validator"""

    def validate_batch(self, inputs: List[Instance], **kwargs: Any) -> None:
        """Takes in a list of Instance objects (each containing their own arg / kwargs) and sets their result flag to true or false"""
        for x in inputs:
            x.result = self._validate(*x.args, **x.kwargs)

    def _validate(
        self,
        api_info: dict,
        question: str,
        answer: str,
        check_arg_question_overlap: bool = True,
        intent_only: bool = False,
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

        for component in sep_components:

            # basic malformedness check
            if any([k for k in component.keys() if k not in [_NAME, _ARGS]]):
                return False

            if intent_only:

                # intent detection should not have arguments
                if component[_ARGS]:
                    return False

                # we'll skip the rest of these checks if we're just interested in intent-detection
                continue

            # since we've checked that each component has an associated api, we can just grab the first (should be identical)
            matching_api = next(
                iter(
                    [api for api in api_info.values() if api[_NAME] == component[_NAME]]
                )
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

                # handle the case where slot values are not mentioned in the question
                if (
                    check_arg_question_overlap
                    and str(arg_content).lower() not in question.lower()
                ):
                    return False

        return True


@register_validator("api_yes_no")
class ApiGenSpecYesNoValidation(APIGenSpecValidator):
    """Class for API Intent Detection Validator"""

    def _validate(self, api_info: dict, question: str, answer: str, **kwargs) -> bool:
        return answer in ["YES", "NO"]
