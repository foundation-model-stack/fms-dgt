# Standard
from typing import Any
import importlib
import os
import re

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.dataloader import BaseDataloader
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.resource import BaseResource
from fms_dgt.blocks.generators.llm import CachingLM, LMGenerator
from fms_dgt.utils import sdg_logger

# TODO: better strategy needed, but this will eliminate some of the confusing errors people get when registering a new class.
REGISTRATION_SEARCHABLE_DIRECTORIES = [
    os.path.join("fms_dgt", "blocks"),
    os.path.join("fms_dgt", "dataloaders"),
    os.path.join("fms_dgt", "datastores"),
]
_REGISTRATION_MODULE_MAP = {}


def _build_importable_registration_map(registration_func: str):
    def extract_registered_classes(file_contents: str):
        classes = []
        for matching_pattern in re.findall(f"{registration_func}\(.*\)", file_contents):
            # last character is ")"
            matching_pattern = matching_pattern.replace(registration_func + "(", "")[
                :-1
            ]
            classes.extend(
                [pattern.replace('"', "") for pattern in matching_pattern.split(",")]
            )
        return classes

    if registration_func not in _REGISTRATION_MODULE_MAP:
        _REGISTRATION_MODULE_MAP[registration_func] = dict()
        for search_dir in REGISTRATION_SEARCHABLE_DIRECTORIES:
            for dirpath, _, filenames in os.walk(search_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if filepath.endswith(".py"):
                        import_path = filepath.replace(os.sep, ".")[:-3]
                        with open(filepath, "r") as f:
                            class_names = extract_registered_classes(f.read())
                            for class_name in class_names:
                                _REGISTRATION_MODULE_MAP[registration_func][
                                    class_name
                                ] = import_path


def _dynamic_import(registration_func: str, class_name: str):
    _build_importable_registration_map(registration_func)
    if (
        registration_func in _REGISTRATION_MODULE_MAP
        and class_name in _REGISTRATION_MODULE_MAP[registration_func]
    ):
        import_path = _REGISTRATION_MODULE_MAP[registration_func][class_name]
        try:
            sdg_logger.info(
                f"Attempting dynamic import of {import_path} for {class_name}"
            )
            importlib.import_module(import_path)
        except ModuleNotFoundError as e:
            # we try both, but we will overwrite with include path
            if f"No module named '{import_path}" not in str(e):
                raise e


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


def get_block(block_name, *args: Any, **kwargs: Any):
    if block_name not in BLOCK_REGISTRY:
        _dynamic_import("register_block", block_name)
    try:
        ret_block = BLOCK_REGISTRY[block_name](*args, **kwargs)
        if isinstance(ret_block, LMGenerator) and "lm_cache" in kwargs:
            ret_block = CachingLM(ret_block, kwargs.get("lm_cache"))
        return ret_block
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
    if resource_name not in RESOURCE_REGISTRY:
        _dynamic_import("register_resource", resource_name)
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


def get_data_builder(name, *args: Any, **kwargs: Any):
    try:
        return DATABUILDER_REGISTRY[name](*args, **kwargs)
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


def get_dataloader(dataloader_name, *args: Any, **kwargs: Any):
    if dataloader_name not in DATALOADER_REGISTRY:
        _dynamic_import("register_dataloader", dataloader_name)
    try:
        return DATALOADER_REGISTRY[dataloader_name](*args, **kwargs)
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


def get_datastore(datastore_name, *args: Any, **kwargs: Any):
    if datastore_name not in DATASTORE_REGISTRY:
        _dynamic_import("register_datastore", datastore_name)
    try:
        return DATASTORE_REGISTRY[datastore_name](*args, **kwargs)
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


def get_task(name, *args: Any, **kwargs: Any):
    try:
        return TASK_REGISTRY[name](*args, **kwargs)
    except KeyError:
        raise ValueError(
            f"Attempted to load task '{name}', but no task for this name found! Supported task names: {', '.join(TASK_REGISTRY.keys())}"
        )
