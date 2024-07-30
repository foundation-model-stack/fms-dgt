# Standard
from pathlib import Path
from typing import Dict, Optional
import os

# Local
from fms_dgt.base.databuilder import DataBuilder
from fms_dgt.base.pipeline import Pipeline
from fms_dgt.base.registry import get_data_builder
from fms_dgt.index import IS_DB_KEY, DataBuilderIndex
import fms_dgt.utils as utils

sdg_logger = utils.sdg_logger


def generate_data(
    data_path: str,
    output_dir: str,
    task_kwargs: Dict,
    builder_kwargs: Dict,
    include_data_path: Optional[str] = None,
    include_config_path: Optional[str] = None,
    include_builder_path: Optional[str] = None,
    restart_generation: bool = False,
):
    # TODO: better naming convention...
    name = (
        Path(os.path.split(data_path)[0]).stem
        if os.path.isfile(data_path)
        else Path(data_path).stem
    )
    output_dir = os.path.join(output_dir, name)

    # check data_path first then seed_tasks_path
    # throw an error if both not found
    # pylint: disable=broad-exception-caught,raise-missing-from
    if data_path and os.path.exists(data_path):
        task_inits = utils.read_data(data_path, include_data_path)
    else:
        raise SystemExit(f"Error: data path ({data_path}) does not exist.")

    # gather data builders here
    builder_list = [t["data_builder"] for t in task_inits]
    builder_index = DataBuilderIndex(
        include_config_path=include_config_path,
        include_builder_path=include_builder_path,
    )
    builder_names = builder_index.match_builders(builder_list)
    sdg_logger.debug("All builders: %s", builder_names)
    for builder in [
        builder for builder in builder_list if builder not in builder_names
    ]:
        if os.path.isfile(builder):
            config = utils.load_yaml_config(builder)
            builder_names.append(config)

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
        builder_names
    ).items():

        # we batch together tasks at the level of data builders
        original_builder_info = builder_index.builder_index[builder_name][0]
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
        if original_builder_info[IS_DB_KEY]:
            # builder_dir is stored in the first builder_info in the list
            utils.import_builder(
                original_builder_info["builder_dir"], include_path=include_builder_path
            )
            data_builder = get_data_builder(builder_name, **all_builder_kwargs)
        else:
            data_builder = Pipeline(**all_builder_kwargs)

        # TODO: ship this off
        data_builder.execute_tasks()

        # TODO: cleanup
        del data_builder
