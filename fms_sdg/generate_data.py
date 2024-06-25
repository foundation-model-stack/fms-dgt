# Standard
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import os
import time

# Third Party
from tqdm import tqdm

# Local
from fms_sdg.base.databuilder import DataBuilder
from fms_sdg.base.registry import get_data_builder
from fms_sdg.base.task import SdgTask
from fms_sdg.databuilders import DataBuilderIndex
import fms_sdg.utils as utils

sdg_logger = utils.sdg_logger


def generate_data(
    max_gen_requests: int,
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

    progress_bar = tqdm(total=len(task_inits), desc="Running generation tasks")
    generate_start = time.time()

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

        # builder_dir is stored in the first builder_info in the list
        utils.import_builder(
            original_builder_info["builder_dir"], include_path=include_builder_path
        )

        data_builder: DataBuilder = get_data_builder(builder_name)(
            config=builder_cfg,
            output_dir=output_dir,
            **builder_kwargs,
        )

        tasks: List[SdgTask] = [
            data_builder.TASK_TYPE(output_dir=output_dir, **task_init, **task_kwargs)
            for task_init in task_inits
            if task_init["data_builder"] == builder_name
        ]

        date_suffix = (
            datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        )
        output_file_discarded = os.path.join(
            output_dir, f"discarded_{name}_{date_suffix}.log"
        )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # load the LM-generated data
        for task in tasks:
            if restart_generation:
                task.clear_data()
            if os.path.exists(task.output_path):
                task.load_data()
                sdg_logger.debug(
                    f"Loaded {len(task.machine_data)} machine-generated data"
                )

        completed_tasks = [task for task in tasks if task.is_complete()]
        tasks = [task for task in tasks if task not in completed_tasks]

        request_idx = 0
        while tasks and request_idx <= max_gen_requests:
            request_idx += 1

            filtered_data = []
            for generated_inst in data_builder.call_with_task_list(request_idx, tasks):
                # save incrementally
                task = next(
                    task for task in tasks if task.name == generated_inst.task_name
                )
                task.save_data(generated_inst)
                filtered_data.append(generated_inst)

            for task in tasks:
                new_data = [
                    gen_inst
                    for gen_inst in filtered_data
                    if gen_inst.task_name == task.name
                ]
                task.machine_data.extend(new_data)
                if task.is_complete():
                    completed_tasks.append(task)
                    progress_bar.update()

            tasks = [task for task in tasks if task not in completed_tasks]

            sdg_logger.info(
                f"Generated {sum([len(task.machine_data) for task in tasks + completed_tasks])} data"
            )

        # TODO: cleanup
        del data_builder

    progress_bar.close()

    generate_duration = time.time() - generate_start
    sdg_logger.info(f"Generation took {generate_duration:.2f}s")
