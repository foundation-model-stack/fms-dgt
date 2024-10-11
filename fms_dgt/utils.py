# Standard
from collections import ChainMap
from pathlib import Path
from typing import Any, Dict, List, TypeVar, Union
import copy
import fnmatch
import importlib.util
import json
import logging
import os

# Third Party
import yaml

log_level = getattr(logging, os.getenv("LOG_LEVEL", "info").upper())
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=log_level,
)
sdg_logger = logging.getLogger("fms_dgt")


def is_module_installed(module_name: str):
    """Checks if a module is installed."""
    result = importlib.util.find_spec(module_name) is not None
    return result


def all_annotations(cls) -> ChainMap:
    return ChainMap(
        *(c.__annotations__ for c in cls.__mro__ if "__annotations__" in c.__dict__)
    )


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def simple_parse_args_string(args_string):
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        k: handle_arg_string(v) for k, v in [arg.split("=") for arg in arg_list]
    }
    return args_dict


def pattern_match(patterns, source_list):
    if isinstance(patterns, str):
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


def import_builder(inp_dir: str) -> None:

    imp_path = inp_dir.replace(os.sep, ".")

    import_path = f"{imp_path}.generate"
    # we try both, but we will overwrite with include path
    try:
        dynamic_import(import_path)
    except ModuleNotFoundError as e:
        # we try both, but we will overwrite with include path
        if f"No module named '{imp_path}" not in str(e):
            raise e


def dynamic_import(import_module: str, throw_top_level_error: bool = False):
    """This function will attempt to import the module specified by `import_module`"""
    try:
        sdg_logger.debug("Attempting dynamic import of %s", import_module)
        importlib.import_module(import_module)
        return True
    except ModuleNotFoundError as e:
        if f"No module named '{import_module}" not in str(e) or throw_top_level_error:
            raise e
        return False


def ignore_constructor(loader, node):
    return node


def import_function(loader, node):
    function_name = loader.construct_scalar(node)
    yaml_path = os.path.dirname(loader.name)

    *module_name, function_name = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)
    module_path = os.path.normpath(os.path.join(yaml_path, "{}.py".format(module_name)))

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function


def load_yaml_config(yaml_path=None, yaml_config=None, yaml_dir=None, mode="full"):
    def load_file(path):
        try:
            if path.endswith(".yaml"):
                data = load_yaml_config(yaml_path=path, mode=mode)
            else:
                with open(path, "r") as f:
                    data = f.read()
            return data
        except Exception as ex:
            # If failed to load, ignore
            raise ex

    def init_include(to_include: Any):
        if type(to_include) == list:
            return [init_include(x) for x in to_include]
        elif type(to_include) == dict:
            return {k: init_include(v) for k, v in to_include.items()}
        elif type(to_include) == str:
            if os.path.isfile(to_include):
                # case where provided include is an absolute path
                add_data = load_file(to_include)
            elif os.path.isfile(os.path.join(yaml_dir, to_include)):
                # case where provided include is a relative path
                add_data = load_file(os.path.join(yaml_dir, to_include))
            else:
                raise ValueError(
                    f"Should not include non-file paths in include directive: {to_include}"
                )
            return add_data
        else:
            raise ValueError(
                f"Unhandled input format in 'include' directive: {to_include}"
            )

    if mode == "simple":
        constructor_fn = ignore_constructor
    elif mode == "full":
        constructor_fn = import_function

    # Add the import_function constructor to the YAML loader
    yaml.add_constructor("!function", constructor_fn)
    if yaml_config is None:
        with open(yaml_path, "rb") as file:
            yaml_config = yaml.full_load(file)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    assert yaml_dir is not None

    if "include" in yaml_config:
        to_include = yaml_config["include"]
        del yaml_config["include"]

        if isinstance(to_include, str):
            to_include = [to_include]

        final_yaml_config = dict()
        to_add = init_include(to_include)
        if type(to_include) == list:
            new_entry = merge_dictionaries(*to_add)
            final_yaml_config.update(new_entry)
        elif type(to_include) == dict:
            final_yaml_config.update(to_add)
        else:
            raise ValueError(
                f"Unhandled input format in 'include' directive: {to_include}"
            )

        final_yaml_config.update(yaml_config)
        return final_yaml_config

    return yaml_config


T = TypeVar("T")


def init_dataclass_from_dict(d_obj: Dict, inp_type: T) -> T:
    if isinstance(d_obj, inp_type):
        return d_obj
    elif type(d_obj) == dict:
        return inp_type(**d_obj)
    elif d_obj is None:
        return inp_type()
    else:
        raise ValueError(
            f"Unhandled input type {type(d_obj)}, cannot convert to type {inp_type}"
        )


def group_data_by_attribute(data_list: List[T], attr: str) -> List[List[T]]:
    attr_values = set([getattr(data_item, attr) for data_item in data_list])
    return [
        [data_item for data_item in data_list if getattr(data_item, attr) == attr_value]
        for attr_value in attr_values
    ]


def merge_dictionaries(*args: List[dict]):
    def _update(d, u):
        for k, v in u.items():
            if k in d and isinstance(d[k], dict) and isinstance(v, dict):
                d[k] = _update(d[k], v)
            else:
                d[k] = v
        return d

    merged_dict = copy.deepcopy(args[0])
    for new_dict in args[1:]:
        _update(merged_dict, new_dict)
    return merged_dict


def get_data_path_name(data_path: str):
    name = (
        Path(os.path.split(data_path)[0]).stem
        if os.path.isfile(data_path)
        else Path(data_path).stem
    )
    return name


def sanitize_path(path: str):
    """
    Sanitize a path against directory traversals
    """
    return os.path.relpath(os.path.normpath(os.path.join(os.sep, path)), os.sep)


# pylint: disable=broad-exception-caught
def read_data_file(file_path: str):
    if file_path.endswith(".yaml"):
        contents = load_yaml_config(file_path)

        if not contents:
            sdg_logger.warning("Skipping %s because it is empty!", file_path)
            return None

        if file_path.startswith("." + os.sep):
            file_path = file_path[len("." + os.sep) :]

        # get seed instruction data
        task = {
            **{
                "data_builder": "simple",
                "created_by": "",
                "seed_examples": [],
            },
            **contents,
        }

        return task


def read_data(data):
    tasks = []
    if os.path.isfile(data):  # data is file
        task = read_data_file(data)
        tasks.append(task)
    else:
        # TODO: fix this once done testing
        for dir, subdirs, files in os.walk(data):
            for file_name in files:
                if file_name == "qna.yaml":
                    file_path = os.path.join(dir, file_name)
                    data = read_data_file(file_path)
                    if data:
                        tasks.append(data)

    return tasks


def load_joint_config(yaml_path: str):

    with open(yaml_path, "r") as f:
        config: dict = yaml.full_load(f)

    data_paths, db_overrides, task_overrides = [], dict(), dict()

    for k, v in config.items():
        if k in ["databuilders", "tasks"]:
            if type(v) != dict:
                raise ValueError(
                    f"'{k}' field in config must be provided as a dictionary where keys are the names of databuilders to override"
                )
            if k == "databuilders":
                db_overrides = v
            else:
                task_overrides = v
        elif k == "task_files":
            if type(v) != list:
                raise ValueError(
                    f"'task_files' field in config must be provided as a list"
                )
            data_paths = v
        else:
            raise ValueError(
                f"Config must only specify 'databuilders' and 'tasks' fields"
            )

    return data_paths, db_overrides, task_overrides


def load_nested_paths(inp: Dict, base_dir: str = None):
    def _is_file(text: str) -> bool:
        return any([text.endswith(ext) for ext in [".json", ".yaml", ".txt"]])

    def _load_file(path: str):
        if path.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)
        elif path.endswith(".yaml"):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        elif path.endswith(".txt"):
            with open(path, "r") as f:
                return str(f.read())
        return path

    def _get_path(fname: str, parent_dir: str):
        if os.path.isfile(fname):
            return os.path.normpath(fname)
        elif parent_dir and os.path.isfile(os.path.join(parent_dir, fname)):
            return os.path.normpath(os.path.join(parent_dir, fname))

    def _pull_paths(d: Union[List, Dict, str], parent_dir: str):
        if isinstance(d, dict):
            for k in d.keys():
                d[k] = _pull_paths(d[k], parent_dir)
        elif isinstance(d, list):
            for i in range(len(d)):
                d[i] = _pull_paths(d[i], parent_dir)
        elif type(d) == str and d and _is_file(d):
            # assigns file_path then checks that file_path is not 'None'
            if (
                file_path := _get_path(d, parent_dir)
            ) not in checked_files and file_path is not None:
                checked_files.add(file_path)
                return _pull_paths(_load_file(file_path), os.path.dirname(file_path))
        return d

    checked_files = set()
    new_dict = _pull_paths(copy.deepcopy(inp), base_dir)

    return new_dict
