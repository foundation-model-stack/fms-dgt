# Standard
from typing import Dict, List
import random

_NAME = "name"
_DESCR = "description"
_PARAMS = "parameters"
_PROPS = "properties"


def api_spec_to_str(
    api_group: Dict,
    pos_functions: List[str],
    task_name: str,
):

    ### APIS_FIELD is the field where we store the string that will be passed to the example generator
    ### this expects
    # API Name: searchJobTitles
    # Description: Search for job titles based on a keyword.
    # Output: A list of job titles, each containing the title name, its unique identifier (ID), and the normalized title.
    # Parameter Name: search
    # Parameter Input: Required. String. A keyword to search for specific job titles.
    ###
    api_infos = list(api_group.items())
    pos_infos = dict([(api_id, api_group[api_id]) for api_id in pos_functions])
    api_infos = list(pos_infos.items())
    if "->parallel_single" in task_name:
        api_infos = [api_infos[0]]
    random.shuffle(api_infos)
    ret_strs = []
    for api_id, api_info in api_infos:
        params = (
            api_info[_PARAMS][_PROPS]
            if _PARAMS in api_info and _PROPS in api_info[_PARAMS]
            else dict()
        )
        arg_str = ", ".join(
            [
                '"' + param_name + '": {"description": "' + param[_DESCR] + '"}'
                for param_name, param in params.items()
            ]
        )
        ret_str = (
            '{"name": "'
            + api_info[_NAME]
            + '", "description": "'
            + api_info[_DESCR]
            + '", "parameters": {'
            + arg_str
            + "}}"
        )
        ret_strs.append(ret_str)
    return "\n".join(ret_strs)
