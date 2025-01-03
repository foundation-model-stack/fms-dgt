# Standard
import json
import os
import subprocess

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.trainers import BaseTrainerBlock
from fms_dgt.blocks.trainers.trainer import TrainingException, make_model_dir
from fms_dgt.constants import DATASET_TYPE
from fms_dgt.utils import get_open_port, sdg_logger

###
# Trainer itself
###

_POST_PROC_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "post_process_adapters_vLLM.py"
)


@register_block("fms-hf-tuning")
class FmsTuningBlock(BaseTrainerBlock):
    def __init__(
        self,
        *args,
        data_formatter_template: str = None,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if data_formatter_template:
            self._training_args["data_formatter_template"] = data_formatter_template
        self._training_args["torch_dtype"] = torch_dtype

    def train(
        self,
        model_id_or_path: str,
        output_dir: str,
        data_to_format: DATASET_TYPE,
        host: str = "0.0.0.0",
        port: int = None,
    ) -> str:

        model_dir = make_model_dir(output_dir)

        if self._run_training(model_dir):

            data_path = os.path.join(output_dir, "dataset", "data.jsonl")
            self.set_dataset(data_to_format, data_path)

            port = get_open_port(host) if port is None else port

            train_cmd = [
                (
                    [
                        "accelerate",
                        "launch",
                        f"--multi_gpu",
                        "--main_process_port",
                        port,
                    ]
                    if self._num_gpus > 1
                    else ["python"]
                ),
                [
                    "-m",
                    "tuning.sft_trainer",
                ],
                ["--model_name_or_path", model_id_or_path],
                ["--training_data_path", data_path],
                ["--output_dir", model_dir],
                ["--save_strategy", "steps"],
                # LORA PARAMS
                ["--use_flash_attn", "true"],
                [
                    "--target_modules",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ],
                ["--lora_alpha", 32],
                ["--lora_dropout", 0.1],
                ["--peft_method", "lora"],
                #
            ] + [[f"--{k}", v] for k, v in self._training_args.items()]
            train_cmd = [str(x) for entry in train_cmd for x in entry]

            post_proc_cmd = [
                "python",
                _POST_PROC_SCRIPT,
                "--model_path",
                model_dir,
                "--output_model_path",
                model_dir,
            ]

            for cmd in [train_cmd, post_proc_cmd]:

                sdg_logger.info(f"Running training command:\n\t{' '.join(cmd)}")

                # run and wait for result
                try:
                    process = subprocess.run(cmd, capture_output=True, text=True)
                    out, err = process.stdout.strip(), process.stderr.strip()
                    if out.strip():
                        sdg_logger.info(out)
                    if err.strip():
                        sdg_logger.error(err)

                    if process.returncode != 0:
                        raise TrainingException(
                            f"Execution failed for command:\n\t{' '.join(cmd)}"
                        )
                except Exception as e:
                    raise e

        if not os.listdir(model_dir):
            raise SystemError(
                f"No checkpoints at model directory {model_dir} were found"
            )

        final_model = "-0"
        for checkpoint in os.listdir(model_dir):
            if checkpoint.startswith("checkpoint") and int(
                checkpoint.split("-")[-1]
            ) >= int(final_model.split("-")[-1]):
                final_model = os.path.join(model_dir, checkpoint)

        # return last model
        return final_model

    def _run_training(self, model_dir: str):

        if os.path.isfile(os.path.join(model_dir, "training_logs.jsonl")):
            with open(os.path.join(model_dir, "training_logs.jsonl"), "r") as jf:
                for line in jf.readlines():
                    pass
                prev_run = json.loads(line)
                return (
                    prev_run["data"]["epoch"] < self._training_args["num_train_epochs"]
                )
        return True

    def release_model(self):
        pass
