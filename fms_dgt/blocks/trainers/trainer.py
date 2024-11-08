###
# Trainer itself
###


# Standard
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import abc
import json
import os

# Third Party
import torch

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.datastore import BaseDatastore
from fms_dgt.utils import sdg_logger


@dataclass
class TrainerData:
    input: str
    output: str

    def to_dict(self):
        return asdict(self)


class BaseTrainerBlock(BaseBlock):
    def __init__(
        self,
        config_path: str = None,
        num_gpus: int = None,
        learning_rate: float = 0.0001,
        logging_steps: int = 100,
        save_steps: int = 50,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        max_steps: int = 100,
        log_level: str = "debug",
        save_total_limit: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize a trainer that trains a model on a dataset input.

        Args:
            config_path (Any): path to config used for trainer
            kwargs (Any): Additional keyword arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self._config_path = config_path

        self._num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus

        training_args = {
            "learning_rate": learning_rate,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_steps": max_steps,
            "save_total_limit": save_total_limit,
            "log_level": log_level,
        }
        self._training_args = {k: v for k, v in training_args.items() if v is not None}

        self._kwargs = kwargs

    def set_dataset(self, datastores: List[BaseDatastore], jsonl_path: str):
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "w") as f:
            for datastore, data_formatter_template in datastores:
                for d in datastore.load_data():
                    f_d = _apply_formatter_template(d, data_formatter_template)
                    f.write(json.dumps(f_d) + "\n")

    def execute(self, *args: Any, **kwargs: Any) -> str:
        return self.train(*args, **kwargs)

    @abc.abstractmethod
    def train(
        self,
        model_id_or_path: str,
        output_dir: str,
        datastores: List[BaseDatastore],
        *args,
        **kwargs,
    ) -> str:
        """Run training and return a model

        Args:
            model_id_or_path (str): Model to initialize from
            output_dir (str): Directory to output model checkpoints
            datastore (BaseDatastore): Datastore that contains all training data
            config_path (Any): path to config used for trainer
            kwargs (Any): Additional keyword arguments to pass to the base class.

        Returns:
            str: Path to model that was trained
        """
        raise NotImplementedError

    def release_model(self):
        pass

    def close(self):
        self.release_model()


class TrainingException(Exception):
    pass


def make_model_dir(output_path: str):
    return os.path.join(output_path, "model")


def _apply_formatter_template(d: Dict, data_formatter_template: Dict):
    ret_dict = dict(data_formatter_template)
    for rd_k, rd_v in ret_dict.items():
        for d_k, d_v in d.items():
            rd_v = rd_v.replace("{{" + d_k + "}}", str(d_v))
        ret_dict[rd_k] = rd_v
    return ret_dict
