# Standard
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
DEFAULT_PROMPT_FILE = "prompt.txt"
DEFAULT_CHUNK_WORD_COUNT = 1000


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--include-builder-path",
        "--include-bp",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are overrides for data builder config files to include.",
    )
    parser.add_argument(
        "--include-data-path",
        "--include-dp",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are overrides for task definition files to include.",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=DEFAULT_PROMPT_FILE,
        metavar="FILE",
        help="Path to prompt file.",
    )
    parser.add_argument(
        "--lm-cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        help="Number of outputs to generate.",
        default=DEFAULT_NUM_OUTPUTS,
    )
    parser.add_argument(
        "--num-prompt-instructions",
        type=int,
        help="Number of prompt instructions to generate.",
        default=DEFAULT_NUM_PROMPT_INSTRUCTIONS,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Path to local data.",
    )
    parser.add_argument(
        "--max-gen-requests",
        type=int,
        help="Maximum number of attempts to solve tasks",
        default=DEFAULT_MAX_GEN_ATTEMPTS,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_GENERATED_FILES_OUTPUT_DIR,
        help="Path to output generated files.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Entirely restart instruction generation.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    python -m fms_sdg.__main__ --data-path <path-to-data> --lm_cache <path-to-cache>
    """

    args = parse_args()

    generate_data(
        num_outputs_to_generate=args.num_outputs,
        num_prompt_instructions=args.num_prompt_instructions,
        data_path=args.data_path,
        max_gen_requests=args.max_gen_requests,
        output_dir=args.output_dir,
        lm_cache=args.lm_cache,
        include_data_path=args.include_data_path,
        include_builder_path=args.include_builder_path,
        prompt_file_path=args.prompt_file,
        restart_generation=args.restart,
    )
