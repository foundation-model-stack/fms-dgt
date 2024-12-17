# Standard
from dataclasses import asdict, dataclass
from typing import Any
import abc
import json
import os

# Third Party
import torch

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.constants import DATASET_TYPE
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
        logging_steps: int = 100,
        save_steps: int = 50,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        max_steps: int = None,
        num_train_epochs: int = 1,
        log_level: str = "debug",
        save_total_limit: int = 1,
        # known good settings
        max_seq_length: int = 4096,
        torch_dtype: str = "bfloat16",
        optim: str = "adamw_torch_fused",
        optim_args: str = "lr=5.0e-5,weight_decay=0.1,eps=1e-10",
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
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_steps": max_steps,
            "num_train_epochs": num_train_epochs,
            "save_total_limit": save_total_limit,
            "log_level": log_level,
            # known good settings
            "max_seq_length": max_seq_length,
            "torch_dtype": torch_dtype,
            "optim": optim,
            "optim_args": optim_args,
        }
        self._training_args = {k: v for k, v in training_args.items() if v is not None}

        self._kwargs = kwargs

    def set_dataset(self, data_to_format: DATASET_TYPE, jsonl_path: str):
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        with open(jsonl_path, "w") as f:
            for d in data_to_format:
                f.write(json.dumps(d) + "\n")

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        return self.execute(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> str:
        return self.train(*args, **kwargs)

    @abc.abstractmethod
    def train(
        self,
        model_id_or_path: str,
        output_dir: str,
        data_to_format: DATASET_TYPE,
        *args,
        **kwargs,
    ) -> str:
        """Run training and return a model

        Args:
            model_id_or_path (str): Model to initialize from
            output_dir (str): Directory to output model checkpoints
            data_to_format (DATASET_TYPE): All training data from one or more tasks
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
