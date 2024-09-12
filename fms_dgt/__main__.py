# Standard
from typing import Optional
import argparse

# Local
from fms_dgt.generate_data import generate_data

DEFAULT_CONFIG = "config.yaml"
MAX_CONTEXT_SIZE = 4096
DEFAULT_NUM_OUTPUTS = 2
DEFAULT_MAX_STALLED_ATTEMPTS = 5
DEFAULT_NUM_PROMPT_INSTRUCTIONS = 2
DEFAULT_MACHINE_BATCH_SIZE = 10
DEFAULT_GENERATED_FILES_OUTPUT_DIR = "output"
DEFAULT_CHUNK_WORD_COUNT = 1000


def get_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    add_base_args(parser)
    add_builder_args(parser)
    add_task_args(parser)

    return parser


def add_base_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("base", "General command-line arguments")
    group.add_argument(
        "--include-builder-paths",
        "--include-bp",
        type=str,
        nargs="*",
        metavar="BUILDER_PATH",
        help="Additional path to include if there are new data builders.",
    )
    group.add_argument(
        "--config-path",
        type=str,
        metavar="CONFIG_PATH",
        help="Path that specifies both data builder configs and tasks.",
    )
    group.add_argument(
        "--data-paths",
        type=str,
        nargs="*",
        help="One or more paths to local data.",
    )
    return group


def add_builder_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("builder", "General command-line arguments")
    group.add_argument(
        "--num-prompt-instructions",
        type=int,
        help="Number of prompt instructions to generate.",
        default=DEFAULT_NUM_PROMPT_INSTRUCTIONS,
    )
    group.add_argument(
        "--prompt-file-path",
        type=str,
        metavar="FILE",
        help="Path to prompt file.",
    )
    group.add_argument(
        "--max-gen-requests",
        type=int,
        help="Maximum number of attempts to solve tasks",
    )
    group.add_argument(
        "--max-stalled-requests",
        type=int,
        help="Maximum number of attempts allowed to solve tasks where no data is produced",
        default=DEFAULT_MAX_STALLED_ATTEMPTS,
    )
    return group


def add_task_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("task", "General command-line arguments")
    group.add_argument(
        "--num-outputs-to-generate",
        type=int,
        help="Number of outputs to generate.",
        default=DEFAULT_NUM_OUTPUTS,
    )
    group.add_argument(
        "--seed-batch-size",
        type=int,
        help="Number of seed examples to pass from data loader to data builder.",
        default=None,
    )
    group.add_argument(
        "--machine-batch-size",
        type=int,
        help="Number of machine generated examples to pass from data loader to data builder.",
        default=DEFAULT_MACHINE_BATCH_SIZE,
    )
    group.add_argument(
        "--restart-generation",
        action="store_true",
        help="Entirely restart instruction generation.",
    )
    group.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_GENERATED_FILES_OUTPUT_DIR,
        help="Path to output generated files.",
    )
    return group


def gather_grouped_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser, group_name: str
):
    for g in parser._action_groups:
        if g.title == group_name:
            kwargs = dict()
            for act in g._group_actions:
                if hasattr(args, act.dest) and getattr(args, act.dest) is not None:
                    kwargs[act.dest] = getattr(args, act.dest)
            return kwargs
    raise ValueError(f"Unrecognized group name: {group_name}")


def parse_cmd_line(arg_list: Optional[list] = None):
    parser = get_parser()

    args = parser.parse_args(arg_list)

    base_args = gather_grouped_args(args, parser, "base")
    builder_kwargs = gather_grouped_args(args, parser, "builder")
    task_kwargs = gather_grouped_args(args, parser, "task")

    return base_args, builder_kwargs, task_kwargs


def main():

    base_args, builder_kwargs, task_kwargs = parse_cmd_line()

    generate_data(
        task_kwargs=task_kwargs,
        builder_kwargs=builder_kwargs,
        **base_args,
    )


if __name__ == "__main__":
    """
    python -m fms_dgt.__main__ --data-paths <path-to-data> --lm_cache <path-to-cache>
    """
    main()
