# Standard
from typing import Dict, List, Mapping, Optional, Union
import collections
import os

# Local
from fms_dgt.constants import BLOCKS_KEY, NAME_KEY
from fms_dgt.utils import sdg_logger
import fms_dgt.utils as utils


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
            if BLOCKS_KEY not in config:
                config[BLOCKS_KEY] = []
            for k in config:
                if k == BLOCKS_KEY and BLOCKS_KEY in override:
                    addlt_blocks = []
                    for block in override[BLOCKS_KEY]:
                        config_block = [
                            i
                            for i, cb in enumerate(config[BLOCKS_KEY])
                            if cb[NAME_KEY] == block[NAME_KEY]
                        ]
                        if config_block:
                            i = config_block[0]
                            config[BLOCKS_KEY][i] = utils.merge_dictionaries(
                                config[BLOCKS_KEY][i], block
                            )
                        else:
                            addlt_blocks.append(block)
                    config[BLOCKS_KEY].extend(addlt_blocks)
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
                f_builder_dir = root.split(os.sep)
                if "databuilders" in f_builder_dir:
                    r_dir_index = f_builder_dir.index("databuilders") - 1
                    f_builder_dir = os.path.join(*f_builder_dir[r_dir_index:])
                    yaml_path = os.path.join(root, f)
                    config = utils.load_yaml_config(yaml_path, mode="simple")

                    # TODO: better validation as to what files are builders and what files aren't
                    if not NAME_KEY in config:
                        return

                    builder_name = config[NAME_KEY]

                    if builder_name in self._builder_index:
                        sdg_logger.warning(
                            "Multiple overriding configuration files detected for data builder %s, the two databuilders are found at '%s' and '%s'",
                            builder_name,
                            f_builder_dir,
                            self._builder_index[builder_name]["builder_dir"],
                        )

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
