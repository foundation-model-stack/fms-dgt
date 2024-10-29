# Standard
import json

# Local
from fms_dgt.blocks.validators.api import APIGenSpecValidator, ApiGenSpecYesNoValidation

single_intent_kwargs = {
    "intent_only": True,
}

multi_intent_kwargs = {
    "intent_only": True,
}

parallel_kwargs = {
    "check_arg_question_overlap": True,
    "intent_only": False,
}

parallel_nested_kwargs = {
    "check_arg_question_overlap": True,
    "intent_only": False,
    "require_nested": True,
}


def get_args(func_calls):
    return {
        k: v for k, v in TEST_APIS.items() if k in [fc["name"] for fc in func_calls]
    }


def test_single_intent():
    validator = APIGenSpecValidator(name="test_single_intent")

    # single intent
    func_calls = [{"name": "add"}]
    question = "add 3 with 4"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **single_intent_kwargs,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(single_intent_kwargs.keys()),
        result_field="result",
    )
    assert test_instance[0]["result"]


def test_multi_intent():
    validator = APIGenSpecValidator(name="test_multi_intent")
    # multiple intent
    func_calls = [
        {"name": "add"},
        {"name": "add_event"},
    ]
    question = "add 3 with 4"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **multi_intent_kwargs,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(multi_intent_kwargs.keys()),
        result_field="result",
    )
    assert test_instance[0]["result"]


def test_parallel_single():
    validator = APIGenSpecValidator(name="test_parallel_single")

    # parallel single
    func_calls = [
        {"name": "add", "arguments": {"n1": 3, "n2": 4}},
        {"name": "add", "arguments": {"n1": 4, "n2": 5}},
    ]
    question = "add 3 with 4 then add 4 with 5"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **parallel_kwargs,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(parallel_kwargs.keys()),
        result_field="result",
    )
    assert test_instance[0]["result"]

    func_calls = [
        {"name": "add", "arguments": {"n1": 3}},
        {"name": "add", "arguments": {"n2": 4}},
    ]
    question = "add 3 with 4"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **parallel_kwargs,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(parallel_kwargs.keys()),
        result_field="result",
    )
    assert not test_instance[0][
        "result"
    ], "Validator should have failed due to required args!"


def test_parallel_multiple():
    validator = APIGenSpecValidator(name="test_parallel_multiple")

    # parallel multiple
    func_calls = [
        {"name": "add", "arguments": {"n1": 3, "n2": 4}},
        {
            "name": "add_event",
            "arguments": {
                "event": "store",
                "info": [{"date": "today", "address": "here"}],
            },
        },
    ]
    question = "add 3 with 4 and add an event store to my calendar"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **parallel_kwargs,
            "check_arg_question_overlap": False,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(parallel_kwargs.keys()) + ["check_arg_question_overlap"],
        result_field="result",
    )
    assert test_instance[0]["result"]

    # parallel multiple invalid args
    func_calls = [
        {"name": "add", "arguments": {"n1": 3, "n2": "store"}},
        {"name": "add_event", "arguments": {"event": "store"}},
    ]
    question = "add 3 with 4 and add an event store to my calendar"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **parallel_kwargs,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(parallel_kwargs.keys()),
        result_field="result",
    )
    assert not test_instance[0]["result"]

    # parallel multiple

    func_calls = [
        {"name": "add", "arguments": {"n1": 3, "n2": 4}},
        {"name": "add_event", "arguments": {"event": "i am going to the store"}},
    ]
    question = "add 3 with 4 and add an event store to my calendar"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **parallel_kwargs,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(parallel_kwargs.keys()),
        result_field="result",
    )
    assert not test_instance[0][
        "result"
    ], "Validator should have failed due to arg content not being in question!"


def test_parallel_nested():
    validator = APIGenSpecValidator(name="test_parallel_nested")

    # parallel multiple
    func_calls = [
        {"name": "add", "arguments": {"n1": 3, "n2": 4}, "label": "$var1"},
        {
            "name": "add_event",
            "arguments": {"event": "$var1.result"},
            "label": "$var2",
        },
    ]
    question = "add 3 with 4 and add an event with the result of the earlier addition to my calendar"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **parallel_nested_kwargs,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(parallel_nested_kwargs.keys()),
        result_field="result",
    )
    assert test_instance[0]["result"]

    func_calls = [
        {"name": "add", "arguments": {"n1": 3, "n2": 4}},
        {"name": "add_event", "arguments": {"event": "i am going to the store"}},
    ]
    question = "add 3 with 4 and add an event store to my calendar"
    api_info = get_args(func_calls)

    test_instance = [
        {
            "a": api_info,
            "b": question,
            "c": json.dumps(func_calls),
            **parallel_nested_kwargs,
        }
    ]
    validator(
        test_instance,
        arg_fields=["a", "b", "c"],
        kwarg_fields=list(parallel_nested_kwargs.keys()),
        result_field="result",
    )
    assert not test_instance[0][
        "result"
    ], "Validator should have failed due to arg content not being in question!"


def test_yes_no():
    validator = ApiGenSpecYesNoValidation(name="test_yes_no")

    for arg_inp in ["YES", "NO", "MAYBE"]:

        test_instance = [
            {
                "a": TEST_APIS,
                "b": "this is a test question",
                "c": arg_inp,
            }
        ]
        validator(
            test_instance,
            arg_fields=["a", "b", "c"],
            result_field="result",
        )
        assert test_instance[0]["result"] == (arg_inp in ["YES", "NO"])


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
        "output_parameters": {"properties": {"result": {"type": "number"}}},
    },
    "add_event": {
        "name": "add_event",
        "parameters": {
            "properties": {
                "event": {"type": "string"},
                "info": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string"},
                            "address": {"type": "string"},
                        },
                        "required": ["date", "address"],
                    },
                },
            },
            "required": ["event"],
        },
        "output_parameters": {"properties": {"added_successfully": {"type": "bool"}}},
    },
}
