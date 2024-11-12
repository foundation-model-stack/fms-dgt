# Standard
from typing import Dict, List, Tuple
import os
import subprocess
import time

# Local
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.trainers import BaseTrainerBlock
from fms_dgt.blocks.trainers.trainer import TrainingException, make_model_dir
from fms_dgt.constants import DATASET_TYPE
from fms_dgt.utils import get_one_line_from_process, sdg_logger

###
# Trainer itself
###


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
        data_to_format: List[Tuple[DATASET_TYPE, Dict]],
    ) -> str:

        model_dir = make_model_dir(output_dir)

        data_path = os.path.join(output_dir, "dataset", "data.jsonl")
        self.set_dataset(data_to_format, data_path)

        cmd = [
            (
                ["accelerate", "launch", f"--multi_gpu"]
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
        ] + [[f"--{k}", v] for k, v in self._training_args.items()]

        cmd = [str(x) for entry in cmd for x in entry]

        sdg_logger.info(f"Starting training with command:\n\t{' '.join(cmd)}")

        # run and wait for result
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            while process.poll() is None:
                # write output from stdout
                while to_write := get_one_line_from_process(process):
                    sdg_logger.info(to_write)
                # sleep so we aren't doing this so frequently
                time.sleep(1)

            # write anything remaining from stdout
            while to_write := get_one_line_from_process(process):
                sdg_logger.info(to_write)

            if process.returncode != 0:
                raise TrainingException(
                    f"Training failed for command:\n\t{' '.join(cmd)}"
                )
            process.kill()
        except Exception as e:
            while to_write := get_one_line_from_process(process):
                sdg_logger.info(to_write)
            process.kill()
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

    def release_model(self):
        pass
