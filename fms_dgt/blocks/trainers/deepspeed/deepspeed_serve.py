# Standard
import json
import os
import re
import subprocess

# Local
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.base.registry import register_block
from fms_dgt.blocks.trainers import BaseTrainerBlock
from fms_dgt.blocks.trainers.trainer import make_model_dir

###
# Trainer itself
###


@register_block("deepspeed")
class DeepspeedTrainerBlock(BaseTrainerBlock):
    def __init__(self, check_interval: int = 10, **kwargs):
        super().__init__(**kwargs)
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
            ["--pid", self._pid],
            ["--check-interval", self._check_interval],
            ["--data-path", data_path],
            ["--config-path", self._config_path],
            ["--model-id-or-path", model_id_or_path],
            ["--output-dir", model_dir],
            [
                "--training-args",
                "'" + json.dumps(self._training_args).replace("'", '"') + "'",
            ],
        ]

        cmd = [str(x) for entry in cmd for x in entry]

        print(" ".join(cmd))
        input("--")

        # run and wait for result
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        final_model = os.path.join(model_dir, "last")

        # return last model
        return final_model
