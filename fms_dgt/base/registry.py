# Standard
from typing import Any

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.dataloader import BaseDataloader
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.resource import BaseResource

BLOCK_REGISTRY = {}


def register_block(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, BaseBlock
            ), f"Block '{name}' ({cls.__name__}) must extend BaseBlock class"

            assert (
                name not in BLOCK_REGISTRY
            ), f"Block named '{name}' conflicts with existing block! Please register with a non-conflicting alias instead."

            BLOCK_REGISTRY[name] = cls
        return cls

    return decorate


def get_block(block_name):
    try:
        return BLOCK_REGISTRY[block_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load block '{block_name}', but no block for this name found! Supported block names: {', '.join(BLOCK_REGISTRY.keys())}"
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


DATASTORE_REGISTRY = {}


def register_datastore(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        for name in names:
            assert issubclass(
                cls, BaseDatastore
            ), f"Datastore '{name}' ({cls.__name__}) must extend BaseDatastore class"

            assert (
                name not in DATASTORE_REGISTRY
            ), f"Datastore named '{name}' conflicts with existing datastore! Please register with a non-conflicting alias instead."

            DATASTORE_REGISTRY[name] = cls
        return cls

    return decorate


def get_datastore(datastore_name):
    try:
        return DATASTORE_REGISTRY[datastore_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load datastore '{datastore_name}', but no datastore for this name found! Supported datastore names: {', '.join(DATASTORE_REGISTRY.keys())}"
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
