# Standard
from typing import Dict, List
import json
import random


def api_spec_to_str(
    api_group: Dict,
    pos_functions: List[str],
    task_name: str,
):
    api_infos = [api_group[api_id] for api_id in pos_functions]
    if "parallel_single" in task_name:
        api_infos = [api_infos[0]]
    random.shuffle(api_infos)
    return "\n".join([json.dumps(api_info, indent=4) for api_info in api_infos])
