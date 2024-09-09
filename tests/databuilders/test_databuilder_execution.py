# Standard
import os
import shutil

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
    ),
    #
    # api function calling
    #
    # ("api", ""),
    #
    # nl2sql
    #
    # ("nl2sql", ""),
]


@pytest.mark.parametrize("data_builder_name,cmd_line_args", to_execute)
def test_data_builders(data_builder_name: str, cmd_line_args: str):

    if os.path.exists(_OUTPUT_DIR):
        shutil.rmtree(_OUTPUT_DIR)

    parser = get_parser()
    arg_list = cmd_line_args.split()
    args = parser.parse_args(arg_list)
    base_args = gather_grouped_args(args, parser, "base")
    builder_kwargs = gather_grouped_args(args, parser, "builder")
    task_kwargs = gather_grouped_args(args, parser, "task")

    generate_data(
        task_kwargs=task_kwargs,
        builder_kwargs=builder_kwargs,
        **base_args,
    )

    if _OUTPUT_DIR in cmd_line_args:
        os.path.exists(_OUTPUT_DIR)
        gen_found = False
        for dirpath, _, fnames in os.walk(_OUTPUT_DIR):
            if any(fstring.startswith("generated_instructions") for fstring in fnames):
                gen_found = True
                break
        assert gen_found, "No instructions file generated"

    if os.path.exists(_OUTPUT_DIR):
        shutil.rmtree(_OUTPUT_DIR)
