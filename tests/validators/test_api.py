# Standard
from typing import List
import json

# Third Party
import pytest

# Local
from fms_sdg.base.instance import Instance
from fms_sdg.validators.api import APIGenSpecValidator, ApiGenSpecYesNoValidation

single_intent_kwargs = {
    "intent_only": True,
    "min_ct": 1,
    "max_ct": 1,
}

multi_intent_kwargs = {
    "intent_only": True,
    "min_ct": 2,
    "max_ct": 2,
}

parallel_kwargs = {
    "check_arg_question_overlap": True,
    "intent_only": False,
    "min_ct": 2,
    "max_ct": 2,
}


def get_args(func_calls):
    return {
        k: v for k, v in TEST_APIS.items() if k in [fc["name"] for fc in func_calls]
    }


class TestApiValidator:
    def test_single_intent(self):
        validator = APIGenSpecValidator("test_single_intent", dict())

        # single intent
        func_calls = [{"name": "add", "arguments": {}}]
        question = "add 3 with 4"
        api_info = get_args(func_calls)
        args = [api_info, question, json.dumps(func_calls)]

        test_instance = [Instance(args, single_intent_kwargs)]
        validator.validate_batch(test_instance)
        assert test_instance[0].result

    def test_multi_intent(self):
        validator = APIGenSpecValidator("test_multi_intent", dict())
        # multiple intent
        func_calls = [
            {"name": "add", "arguments": {}},
            {"name": "add_event", "arguments": {}},
        ]
        question = "add 3 with 4"
        api_info = get_args(func_calls)
        args = [api_info, question, json.dumps(func_calls)]

        test_instance = [Instance(args, multi_intent_kwargs)]
        validator.validate_batch(test_instance)
        assert test_instance[0].result

    def test_parallel_single(self):
        validator = APIGenSpecValidator("test_parallel_single", dict())

        # parallel single
        func_calls = [
            {"name": "add", "arguments": {"n1": 3, "n2": 4}},
            {"name": "add", "arguments": {"n1": 4, "n2": 5}},
        ]
        question = "add 3 with 4 then add 4 with 5"
        api_info = get_args(func_calls)
        args = [api_info, question, json.dumps(func_calls)]

        test_instance = [Instance(args, parallel_kwargs)]
        validator.validate_batch(test_instance)
        assert test_instance[0].result

        func_calls = [
            {"name": "add", "arguments": {"n1": 3}},
            {"name": "add", "arguments": {"n2": 4}},
        ]
        question = "add 3 with 4"
        api_info = get_args(func_calls)
        args = [api_info, question, json.dumps(func_calls)]

        test_instance = [Instance(args, parallel_kwargs)]
        validator.validate_batch(test_instance)
        assert not test_instance[
            0
        ].result, "Validator should have failed due to required args!"

    def test_parallel_multiple(self):
        validator = APIGenSpecValidator("test_parallel_multiple", dict())

        # parallel multiple
        func_calls = [
            {"name": "add", "arguments": {"n1": 3, "n2": 4}},
            {"name": "add_event", "arguments": {"event": "store"}},
        ]
        question = "add 3 with 4 and add an event store to my calendar"
        api_info = get_args(func_calls)
        args = [api_info, question, json.dumps(func_calls)]

        test_instance = [Instance(args, parallel_kwargs)]
        validator.validate_batch(test_instance)
        assert test_instance[0].result

        func_calls = [
            {"name": "add", "arguments": {"n1": 3, "n2": 4}},
            {"name": "add_event", "arguments": {"event": "i am going to the store"}},
        ]
        question = "add 3 with 4 and add an event store to my calendar"
        api_info = get_args(func_calls)
        args = [api_info, question, json.dumps(func_calls)]

        test_instance = [Instance(args, parallel_kwargs)]
        validator.validate_batch(test_instance)
        assert not test_instance[
            0
        ].result, (
            "Validator should have failed due to arg content not being in question!"
        )

    def test_yes_no(self):
        validator = ApiGenSpecYesNoValidation("test_yes_no", dict())

        for arg_inp in ["YES", "NO", "MAYBE"]:
            args = [TEST_APIS, "this is a test question", arg_inp]
            test_instance = [Instance(args)]
            validator.validate_batch(test_instance)
            assert test_instance[0].result == (arg_inp in ["YES", "NO"])


TEST_APIS = {
    "add": {
        "name": "add",
        "parameters": {
            "properties": {
                "n1": {"type": "number"},
                "n2": {"type": "number"},
            },
            "required": ["n1", "n2"],
        },
    },
    "add_event": {
        "name": "add_event",
        "parameters": {
            "properties": {"event": {"type": "string"}},
            "required": ["event"],
        },
    },
}
