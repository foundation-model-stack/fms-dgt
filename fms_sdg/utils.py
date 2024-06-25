# Standard
from collections import ChainMap
from typing import Any, Callable, List, TypeVar
import collections
import copy
import fnmatch
import importlib.util
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
sdg_logger = logging.getLogger("fms_sdg")


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


def import_builder(inp_dir: str, include_path: str = None) -> None:
    if include_path is not None:
        include_path = include_path.replace(os.sep, ".")

    loaded = False

    # TODO: this must be generalized
    for imp_path in ["fms_sdg.databuilders", include_path]:
        if imp_path is not None:
            import_path = f"{imp_path}.{inp_dir}.generate"
            try:
                importlib.import_module(import_path)
                loaded = True
            except ModuleNotFoundError as e:
                # we try both, but we will overwrite with include path
                if f"No module named '{imp_path}" not in str(e):
                    raise e

    if not loaded:
        err_str = f"No module named 'fms_sdg.databuilders.{inp_dir}.generate'"
        if include_path is not None:
            err_str += f" or '{include_path}.{inp_dir}.generate'"
        raise ModuleNotFoundError(err_str)


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
            for entry in to_add:
                final_yaml_config.update(entry)
        elif type(to_include) == dict:
            final_yaml_config.update(to_add)
        else:
            raise ValueError(
                f"Unhandled input format in 'include' directive: {to_include}"
            )

        final_yaml_config.update(yaml_config)
        return final_yaml_config

    return yaml_config


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


class Reorderer:
    def __init__(self, arr: List[Any], fn: Callable) -> None:
        """Reorder an array according to some function

        Args:
            arr (List[Any]): The initial array
            fn (Callable[[Any], Any]): A function to determine the priority of elements
        """
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr.sort(key=lambda x: fn(x[1]))
        self.arr = arr

    def get_reordered(self):
        """Gets the reordered array

        Returns:
            List[Any]: The reordered array
        """
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        """Restores the original order of a new array based on the old array's order

        Args:
            newarr (List[Any]): The array to be restored

        Returns:
            List[Any]: The array restored to the original order
        """
        res = [None] * self.size
        cov = [False] * self.size

        for (ind, _), v in zip(self.arr, newarr):
            res[ind] = v
            cov[ind] = True

        assert all(cov)

        return res


T = TypeVar("T")


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


# pylint: disable=broad-exception-caught
def read_data_file(file_path: str):
    if file_path.endswith(".yaml"):
        contents = load_yaml_config(file_path)

        if not contents:
            sdg_logger.warn(f"Skipping {file_path} because it is empty!")
            return None

        if file_path.startswith("." + os.sep):
            file_path = file_path[len("." + os.sep) :]

        # get seed instruction data
        task_name = "->".join(os.path.dirname(file_path).split(os.sep))
        data_builder = contents.get("data_builder", "simple")
        created_by = contents.get("created_by", "")
        seed_data = contents.get("seed_examples", [dict()])
        task = {
            **{
                "name": task_name,
                "data_builder": data_builder,
                "created_by": created_by,
                "seed_data": seed_data,
            },
            **contents,
        }

        return task


def read_data(data, include_data_path=None):
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

    if include_data_path is not None:
        tasks = override_data_configs(include_data_path, tasks)

    return tasks


def override_data_configs(include_data_path, tasks):
    override_with = read_data_override_files(include_data_path)
    assert all(
        ["data_path" in ow for ow in override_with]
    ), f"Must specify data path to override in files!"
    # general precedence will be for subsets to be overwritten first
    override_with = sorted(override_with, key=lambda x: len(x["data_path"]))
    for ow in override_with:
        assert (
            "data_path" in ow
        ), f"Must specify data path to override in file: {ow['file_path']}"
        tax_path = ow["data_path"]
        ow = {k: v for k, v in ow.items() if k not in ["file_path", "data_path"]}
        for i in range(len(tasks)):
            if tasks[i]["data_path"].startswith(tax_path):
                tasks[i] = merge_dictionaries(tasks[i], ow)
    return tasks


def read_data_override_files(data):
    overrides = []
    assert os.path.exists(data), f"Path to override data files does not exist: {data}"
    if os.path.isfile(data):
        config = load_yaml_config(data)
        config["file_path"] = data
        overrides.append(config)
    else:
        for dir, subdirs, files in os.walk(data):
            for file_name in files:
                if file_name == "qna.yaml":
                    file_path = os.path.join(dir, file_name)
                    config = load_yaml_config(file_path)
                    config["file_path"] = file_path
                    overrides.append(config)
    return overrides
