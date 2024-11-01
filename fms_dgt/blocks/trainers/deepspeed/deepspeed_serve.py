# Standard
import json
import os
import subprocess
import time

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.trainers import BaseTrainerBlock
from fms_dgt.blocks.trainers.deepspeed.train import TrainingException
from fms_dgt.blocks.trainers.trainer import make_model_dir
from fms_dgt.utils import sdg_logger

###
# Trainer itself
###


@register_block("deepspeed")
class DeepspeedTrainerBlock(BaseTrainerBlock):
    def __init__(self, *args, check_interval: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self._pid = os.getpid()
        self._check_interval = check_interval

    def train(
        self,
        model_id_or_path: str,
        output_dir: str,
        datastore: BaseDatastore,
    ) -> str:

        model_dir = make_model_dir(output_dir)

        data_path = os.path.join(output_dir, "dataset")
        self.set_dataset(datastore, data_path)

        cmd = [
            [
                "deepspeed",
                f"--num_gpus={self._num_gpus}",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py"),
            ],
            ["--data-path", data_path],
            ["--config-path", self._config_path],
            ["--model-id-or-path", model_id_or_path],
            ["--output-dir", model_dir],
            [
                "--training-args",
                json.dumps(self._training_args).replace("'", '"'),
            ],
        ]

        cmd = [str(x) for entry in cmd for x in entry]

        sdg_logger.info(f"Starting training with command:\n\t{' '.join(cmd)}")

        # run and wait for result
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            while process.poll() is None:
                for proc in [process.stdout, process.stderr]:
                    sdg_logger.info(proc.readline().decode("utf-8").strip())
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
        except Exception as e:
            process.kill()
            raise e

        final_model = os.path.join(model_dir, "last")

        # return last model
        return final_model
