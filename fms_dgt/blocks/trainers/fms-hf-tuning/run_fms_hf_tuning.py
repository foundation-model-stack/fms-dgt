# Standard
from typing import List
import os
import subprocess
import time

# Third Party
import psutil

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.trainers import BaseTrainerBlock
from fms_dgt.blocks.trainers.trainer import TrainingException, make_model_dir
from fms_dgt.utils import sdg_logger

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
        datastores: List[BaseDatastore],
    ) -> str:

        model_dir = make_model_dir(output_dir)

        data_path = os.path.join(output_dir, "dataset", "data.jsonl")
        self.set_dataset(datastores, data_path)

        cmd = [
            [
                ("accelerate launch" if self._num_gpus > 1 else "python"),
                "-m",
                "tuning.sft_trainer",
            ],
            ["--model_name_or_path", model_id_or_path],
            ["--training_data_path", data_path],
            ["--output-dir", model_dir],
        ] + [[f"--{k}", v] for k, v in self._training_args.items()]

        cmd = [str(x) for entry in cmd for x in entry]

        sdg_logger.info(f"Starting training with command:\n\t{' '.join(cmd)}")

        # run and wait for result
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            while process.poll() is None:
                for proc in [process.stdout, process.stderr]:
                    sdg_logger.info(proc.read().decode("utf-8").strip())
                time.sleep(1)
            out, err = (
                process.stdout.read().decode("utf-8").strip(),
                process.stderr.read().decode("utf-8").strip(),
            )
            if out:
                sdg_logger.info(out)
            if err:
                sdg_logger.error(err)
            if process.returncode != 0:
                raise TrainingException(
                    f"Training failed for command:\n\t{' '.join(cmd)}"
                )
            process.kill()
        except Exception as e:
            process.kill()
            raise e

        final_model = os.path.join(model_dir, "last")

        # return last model
        return final_model

    def release_model(self):
        pass
