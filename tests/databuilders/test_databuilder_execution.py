# Standard
import gc
import multiprocessing
import os
import shutil
import time

# Third Party
import pytest

# Local
from fms_dgt.__main__ import *

_BASE_REPO_PATH = os.path.split(
    os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
)[0]
_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_outputs")

to_execute = [
    #
    # simple
    #
    (
        "simple",
        f"--data-paths {os.path.join(_BASE_REPO_PATH, 'data/generation/logical_reasoning/causal/qna.yaml')} --num-outputs-to-generate 1 --output-dir {_OUTPUT_DIR}",
        25,
    ),
    #
    # api function calling
    #
    (
        "api",
        f"--data-paths {os.path.join(_BASE_REPO_PATH, 'data/generation/code/apis/glaive/detection/single_api/qna.yaml')} --num-outputs-to-generate 1 --output-dir {_OUTPUT_DIR}",
        25,
    ),
    # #
    # # nl2sql
    # #
    (
        "nl2sql",
        f"--data-paths {os.path.join(_BASE_REPO_PATH, 'data/generation/code/sql/nl2sql/orders/qna.yaml')} --num-outputs-to-generate 1 --output-dir {_OUTPUT_DIR}",
        25,
    ),
]


# def gen_data(task_kwargs, builder_kwargs, base_args):
#     generate_data(
#         task_kwargs=task_kwargs,
#         builder_kwargs=builder_kwargs,
#         **base_args,
#     )


@pytest.mark.parametrize("data_builder_name,cmd_line_args,timeout", to_execute)
def test_data_builders(data_builder_name: str, cmd_line_args: str, timeout: int):
    """This file contains execution tests for each data builder (in the same way it would be called from the command-line). To add a new test,
    add your data builder, its command-line arguments, and a timeout to the 'to_execute' list. The command line arguments should result in a
    reasonably quick execution and the timeout value should indicate the maximum allowable time it takes to run the command.

    NOTE: this assumes the default settings of your databuilder config are what you want to use for testing

    Args:
        data_builder_name (str): name of databuilder to be tested
        cmd_line_args (str): command-line argument string
        timeout (int): time in seconds to allocate to test
    """
    if os.path.exists(_OUTPUT_DIR):
        shutil.rmtree(_OUTPUT_DIR)

    parser = get_parser()
    arg_list = cmd_line_args.split()
    args = parser.parse_args(arg_list)
    base_args = gather_grouped_args(args, parser, "base")
    builder_kwargs = gather_grouped_args(args, parser, "builder")
    task_kwargs = gather_grouped_args(args, parser, "task")

    p = multiprocessing.Process(
        target=generate_data, args=(task_kwargs, builder_kwargs), kwargs=base_args
    )

    p.start()

    # wait for 'timeout' seconds or until process finishes
    p.join(timeout)

    assert (
        not p.is_alive()
    ), f"'{data_builder_name}' data builder took to long to execute"

    # if thread is still active
    if p.is_alive():
        p.terminate()
        time.sleep(1)
        if p.is_alive():
            p.kill()
            time.sleep(1)
        gc.collect()

    time.sleep(5)

    if _OUTPUT_DIR in cmd_line_args:
        os.path.exists(_OUTPUT_DIR)
        gen_found = False
        for _, _, fnames in os.walk(_OUTPUT_DIR):
            if any(fstring.startswith("outputs") for fstring in fnames):
                gen_found = True
                break
        assert (
            gen_found
        ), f"No instructions file generated for '{data_builder_name}' data builder"

    if os.path.exists(_OUTPUT_DIR):
        shutil.rmtree(_OUTPUT_DIR)
