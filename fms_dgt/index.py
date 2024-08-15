# Standard
from typing import Dict, List, Mapping, Optional, Union
import collections
import os

# Local
from fms_dgt.utils import sdg_logger
import fms_dgt.utils as utils

_BLOCKS_KEY, _NAME_KEY = "blocks", "name"


class DataBuilderIndex:
    """DataBuilderIndex indexes all data builders from the default `fms_dgt/databuilders/` and an optional directory if provided."""

    def __init__(self, include_builder_paths: Optional[List[str]] = None) -> None:
        include_builder_paths = include_builder_paths if include_builder_paths else []
        self._builder_index = collections.defaultdict(list)
        self._initialize_data_builders(include_paths=include_builder_paths)

        self._all_builders = sorted(list(self._builder_index.keys()))
        self.data_builder_group_map = collections.defaultdict(list)

    def _initialize_data_builders(self, include_paths: List[str]):
        # default location for databuilders / pipelines
        all_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), comp)
            for comp in ["databuilders", "pipelines"]
        ]
        # additional paths for databuilders / pipelines
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

    def _load_individual_builder_config(
        self,
        name: Optional[str] = None,
        config_overrides: Optional[Dict] = None,
    ) -> Mapping:
        def override_builder_config(config: Dict, override: Dict):
            for k in config:
                if k == _BLOCKS_KEY and _BLOCKS_KEY in override:
                    for i in range(len(config[_BLOCKS_KEY])):
                        block = config[_BLOCKS_KEY][i]
                        override_blocks = [
                            ob
                            for ob in override[_BLOCKS_KEY]
                            if ob[_NAME_KEY] == block[_NAME_KEY]
                        ]
                        if override_blocks:
                            config[_BLOCKS_KEY][i] = utils.merge_dictionaries(
                                block, *override_blocks
                            )
                else:
                    config[k] = utils.merge_dictionaries(
                        config[k], override.get(k, dict())
                    )

        if config_overrides is None:
            config_overrides = dict()

        config = self.builder_index[name]["config"]
        override = config_overrides.get(name, dict())
        override_builder_config(config, override)

        return (name, config)

    def load_builder_configs(
        self,
        builder_list: Optional[Union[str, list]] = None,
        config_overrides: Optional[Dict] = None,
    ) -> dict:
        if isinstance(builder_list, str):
            builder_list = [builder_list]

        all_loaded_builders = dict(
            [
                self._load_individual_builder_config(builder, config_overrides)
                for builder in builder_list
            ]
        )

        return all_loaded_builders

    def _get_data_builder(self, builder_path: str):
        def add_file(root, f):

            if f.endswith(".yaml"):
                f_builder_dir = os.path.split(root)[-1]
                yaml_path = os.path.join(root, f)
                config = utils.load_yaml_config(yaml_path, mode="simple")

                # TODO: better validation as to what files are builders and what files aren't
                if not _NAME_KEY in config:
                    return

                builder_name = config[_NAME_KEY]

                assert (
                    builder_name not in self._builder_index
                ), f"Multiple overriding configuration files detected for data builder {builder_name}"

                self._builder_index[builder_name] = {
                    "builder_dir": f_builder_dir,
                    "config": config,
                }

        if os.path.isfile(builder_path):
            root, f = os.path.split(builder_path)
            add_file(root, f)
        else:
            for root, _, file_list in os.walk(builder_path):
                for f in file_list:
                    add_file(root, f)
