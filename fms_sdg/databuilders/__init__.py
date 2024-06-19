# Standard
from functools import partial
from typing import Dict, List, Mapping, Optional, Union
import collections
import os

# Local
from fms_sdg.utils import sdg_logger
import fms_sdg.utils as utils


class DataBuilderIndex:
    """DataBuilderIndex indexes all data builders from the default `fms_sdg/databuilders/` and an optional directory if provided."""

    def __init__(
        self,
        include_builder_path: Optional[str] = None,
        include_config_path: Optional[str] = None,
    ) -> None:
        self._builder_index = collections.defaultdict(list)
        self._initialize_data_builders(
            include_paths=[include_builder_path, include_config_path]
        )

        self._all_builders = sorted(list(self._builder_index.keys()))
        self.data_builder_group_map = collections.defaultdict(list)

    def _initialize_data_builders(self, include_paths: List[str]):
        all_paths = [os.path.dirname(os.path.abspath(__file__)) + "/"]
        for to_include in include_paths:
            if to_include is not None:
                if isinstance(to_include, str):
                    to_include = [to_include]
                all_paths.extend(to_include)

        for data_builder_dir in all_paths:
            self._get_data_builder(data_builder_dir)

    @property
    def all_builders(self):
        return self._all_builders

    @property
    def builder_index(self):
        return self._builder_index

    def match_builders(self, builder_list):
        return utils.pattern_match(builder_list, self.all_builders)

    def _name_is_registered(self, name) -> bool:
        if name in self.all_builders:
            return True
        return False

    def _get_yaml_path(self, name):
        if name not in self.builder_index:
            raise ValueError
        return self.builder_index[name]["yaml_path"]

    def _get_config(self, name):
        if name not in self.builder_index:
            raise ValueError
        yaml_paths = [cd["yaml_path"] for cd in self.builder_index[name]]
        return (
            [utils.load_yaml_config(yp, mode="full") for yp in yaml_paths]
            if yaml_paths
            else dict()
        )

    def _load_individual_builder_config(
        self,
        name_or_config: Optional[Union[str, dict]] = None,
    ) -> Mapping:
        def load_builder(config, builder_name, yaml_path=None):
            if "include" in config:
                if yaml_path is None:
                    raise ValueError
                config.update(
                    utils.load_yaml_config(
                        yaml_path,
                        yaml_config={"include": config.pop("include")},
                        mode="full",
                    )
                )
            return {builder_name: config}

        configs = self._get_config(name_or_config)
        merged_config = utils.merge_dictionaries(*configs)
        return load_builder(merged_config, builder_name=name_or_config)

    def load_builder_configs(
        self, builder_list: Optional[Union[str, list]] = None
    ) -> dict:
        if isinstance(builder_list, str):
            builder_list = [builder_list]

        all_loaded_builders = dict(
            collections.ChainMap(
                *map(self._load_individual_builder_config, builder_list)
            )
        )
        return all_loaded_builders

    def _get_data_builder(self, builder_path: str):
        def add_file(root, f):
            if f.endswith(".yaml"):
                f_builder_dir = os.path.split(root)[-1]
                yaml_path = os.path.join(root, f)
                config = utils.load_yaml_config(yaml_path, mode="simple")

                # TODO: Clean this up and get rid of it, just load top two levels
                if not "name" in config:
                    return

                builder_name = config["name"]

                self._builder_index[builder_name].append(
                    {
                        "yaml_path": yaml_path,
                        "builder_dir": f_builder_dir,
                    }
                )

                if len(self._builder_index[builder_name]) > 2:
                    sdg_logger.warning(
                        f"Multiple overriding configuration files detected for data builder {builder_name}. By default, configurations will be merged."
                    )

        if os.path.isfile(builder_path):
            root, f = os.path.split(builder_path)
            add_file(root, f)
        else:
            for root, _, file_list in os.walk(builder_path):
                for f in file_list:
                    add_file(root, f)
