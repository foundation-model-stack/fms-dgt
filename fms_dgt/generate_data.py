# Standard
from typing import Dict, List, Optional
import os

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.registry import get_data_builder
from fms_dgt.index import DataBuilderIndex
import fms_dgt.utils as utils

sdg_logger = utils.sdg_logger


def generate_data(
    output_dir: str,
    task_kwargs: Dict,
    builder_kwargs: Dict,
    data_paths: Optional[List[str]] = None,
    config_path: Optional[List[str]] = None,
    include_builder_paths: Optional[List[str]] = None,
    restart_generation: bool = False,
):

    if data_paths and config_path:
        raise ValueError(
            f"Specify only one of ['data-paths', 'config-path'] in the arguments, not both"
        )

    if config_path:
        data_paths, config_overrides = utils.load_joint_config(config_path)
    elif data_paths:
        data_paths, config_overrides = data_paths, None
    else:
        raise ValueError(
            f"One of ['data-paths', 'config-path'] must be provided in the arguments"
        )

    # TODO: better naming convention...
    names = []
    for data_path in data_paths:
        names.append(utils.get_data_path_name(data_path))
    name = "_AND_".join(names)
    output_dir = os.path.join(output_dir, name)

    # check data_path first then seed_tasks_path
    # throw an error if both not found
    # pylint: disable=broad-exception-caught,raise-missing-from
    task_inits = []
    for data_path in data_paths:
        if data_path and os.path.exists(data_path):
            task_inits.extend(utils.read_data(data_path))
        else:
            raise SystemExit(f"Error: data path ({data_path}) does not exist.")

    # gather data builders here
    builder_list = [t["data_builder"] for t in task_inits]
    builder_index = DataBuilderIndex(
        include_builder_paths=include_builder_paths,
    )
    builder_names = builder_index.match_builders(builder_list)
    sdg_logger.debug("All builders: %s", builder_names)
    builder_missing = set(
        [
            builder
            for builder in builder_list
            if builder not in builder_names and "*" not in builder
        ]
    )

    if builder_missing:
        missing = ", ".join(builder_missing)
        raise ValueError(f"Builder specifications not found: [{missing}]")

    for builder_name, builder_cfg in builder_index.load_builder_configs(
        builder_names, config_overrides=config_overrides
    ).items():

        # we batch together tasks at the level of data builders
        original_builder_info = builder_index.builder_index[builder_name]
        if isinstance(builder_cfg, tuple):
            _, builder_cfg = builder_cfg
            if builder_cfg is None:
                continue

        sdg_logger.debug("Builder config for %s: %s", builder_name, builder_cfg)

        all_builder_kwargs = {
            "config": builder_cfg,
            "output_dir": output_dir,
            "restart_generation": restart_generation,
            "task_inits": [
                task_init
                for task_init in task_inits
                if task_init["data_builder"] == builder_name
            ],
            "task_kwargs": task_kwargs,
            **builder_kwargs,
        }

        try:
            # first see if databuilder is loaded by default
            data_builder: DataBuilder = get_data_builder(
                builder_name, **all_builder_kwargs
            )
        except ValueError as e:
            if f"Attempted to load data builder '{builder_name}'" in str(e):
                utils.import_builder(
                    original_builder_info["builder_dir"],
                    include_paths=include_builder_paths,
                )
                data_builder: DataBuilder = get_data_builder(
                    builder_name, **all_builder_kwargs
                )
            else:
                raise e

        # TODO: ship this off
        data_builder.execute_tasks()

        # TODO: cleanup
        del data_builder
