# Standard
from typing import List
import argparse

# Local
from fms_sdg.generate_data import generate_data

DEFAULT_CONFIG = "config.yaml"
DEFAULT_DATA_PATH = "data"
MAX_CONTEXT_SIZE = 4096
DEFAULT_NUM_OUTPUTS = 2
DEFAULT_MAX_GEN_ATTEMPTS = 2
DEFAULT_NUM_PROMPT_INSTRUCTIONS = 2
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
        "--include-builder-path",
        "--include-bp",
        type=str,
        metavar="DIR",
        help="Additional path to include if there are new data builders.",
    )
    group.add_argument(
        "--include-config-path",
        type=str,
        metavar="DIR",
        help="Additional path to include if there are overrides for data builder config files.",
    )
    group.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to local data.",
    )
    group.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_GENERATED_FILES_OUTPUT_DIR,
        help="Path to output generated files.",
    )
    group.add_argument(
        "--restart-generation",
        action="store_true",
        help="Entirely restart instruction generation.",
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
        "--lm-cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    group.add_argument(
        "--max-gen-requests",
        type=int,
        help="Maximum number of attempts to solve tasks",
        default=DEFAULT_MAX_GEN_ATTEMPTS,
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


def main():
    parser = get_parser()

    args = parser.parse_args()

    base_args = gather_grouped_args(args, parser, "base")
    builder_kwargs = gather_grouped_args(args, parser, "builder")
    task_kwargs = gather_grouped_args(args, parser, "task")

    generate_data(
        task_kwargs=task_kwargs,
        builder_kwargs=builder_kwargs,
        **base_args,
    )


if __name__ == "__main__":
    """
    python -m fms_sdg.__main__ --data-path <path-to-data> --lm_cache <path-to-cache>
    """
    main()
