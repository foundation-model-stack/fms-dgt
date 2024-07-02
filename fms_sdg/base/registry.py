# Standard
from typing import Any
import logging

# Local
from fms_sdg.base.dataloader import BaseDataloader
from fms_sdg.base.generator import BaseGenerator
from fms_sdg.base.resource import BaseResource
from fms_sdg.base.validator import BaseValidator

GENERATOR_REGISTRY = {}


def register_generator(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, BaseGenerator
            ), f"Generator '{name}' ({cls.__name__}) must extend BaseGenerator class"

            assert (
                name not in GENERATOR_REGISTRY
            ), f"Generator named '{name}' conflicts with existing generator! Please register with a non-conflicting alias instead."

            GENERATOR_REGISTRY[name] = cls
        return cls

    return decorate


def get_generator(model_name):
    try:
        return GENERATOR_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load generator '{model_name}', but no generator for this name found! Supported generator names: {', '.join(GENERATOR_REGISTRY.keys())}"
        )


VALIDATOR_REGISTRY = {}


def register_validator(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, BaseValidator
            ), f"Validator '{name}' ({cls.__name__}) must extend BaseValidator class"

            assert (
                name not in VALIDATOR_REGISTRY
            ), f"Validator named '{name}' conflicts with existing validator! Please register with a non-conflicting alias instead."

            VALIDATOR_REGISTRY[name] = cls
        return cls

    return decorate


def get_validator(model_name):
    try:
        return VALIDATOR_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load validator '{model_name}', but no validator for this name found! Supported validator names: {', '.join(VALIDATOR_REGISTRY.keys())}"
        )


RESOURCE_REGISTRY = {}
RESOURCE_OBJECTS = {}


def register_resource(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, BaseResource
            ), f"Resource '{name}' ({cls.__name__}) must extend BaseResource class"

            assert (
                name not in RESOURCE_REGISTRY
            ), f"Resource named '{name}' conflicts with existing resource! Please register with a non-conflicting alias instead."

            RESOURCE_REGISTRY[name] = cls
        return cls

    return decorate


def get_resource(resource_name, *args: Any, **kwargs: Any):
    try:
        resource: BaseResource = RESOURCE_REGISTRY[resource_name](*args, **kwargs)
    except KeyError:
        raise ValueError(
            f"Attempted to load resource '{resource_name}', but no resource for this name found! Supported resource names: {', '.join(RESOURCE_REGISTRY.keys())}"
        )
    if resource.id not in RESOURCE_OBJECTS:
        RESOURCE_OBJECTS[resource.id] = resource
    return RESOURCE_OBJECTS[resource.id]


DATABUILDER_REGISTRY = {}
ALL_DATABUILDERS = set()


def register_data_builder(name):
    def decorate(fn):
        assert (
            name not in DATABUILDER_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        DATABUILDER_REGISTRY[name] = fn
        ALL_DATABUILDERS.add(name)
        return fn

    return decorate


def get_data_builder(name):
    try:
        return DATABUILDER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to load data builder '{name}', but no data builder for this name found! Supported data builder names: {', '.join(DATABUILDER_REGISTRY.keys())}"
        )


DATALOADER_REGISTRY = {}


def register_dataloader(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, BaseDataloader
            ), f"Dataloader '{name}' ({cls.__name__}) must extend BaseDataloader class"

            assert (
                name not in DATALOADER_REGISTRY
            ), f"Dataloader named '{name}' conflicts with existing dataloader! Please register with a non-conflicting alias instead."

            DATALOADER_REGISTRY[name] = cls
        return cls

    return decorate


def get_dataloader(dataloader_name):
    try:
        return DATALOADER_REGISTRY[dataloader_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load dataloader '{dataloader_name}', but no dataloader for this name found! Supported dataloader names: {', '.join(DATALOADER_REGISTRY.keys())}"
        )


TASK_REGISTRY = {}
ALL_TASKS = set()


def register_task(name):
    def decorate(fn):
        assert (
            name not in TASK_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        TASK_REGISTRY[name] = fn
        ALL_TASKS.add(name)
        return fn

    return decorate


def get_task(name):
    try:
        return TASK_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to load task '{name}', but no task for this name found! Supported task names: {', '.join(TASK_REGISTRY.keys())}"
        )
