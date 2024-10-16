###
# Trainer itself
###


# Standard
from dataclasses import asdict, dataclass
from typing import Any
import abc
import os

# Third Party
from datasets import Dataset
import torch

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.datastore import BaseDatastore


@dataclass
class TrainerData:
    input: str
    output: str

    def to_dict(self):
        return asdict(self)


class BaseTrainerBlock(BaseBlock):
    def __init__(
        self,
        config_path: str,
        num_gpus: int = None,
        learning_rate: float = 0.0001,
        fp16: bool = True,
        logging_steps: int = 100,
        save_steps: int = 50,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        max_steps: int = 100,
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
            "fp16": fp16,
            "logging_steps": logging_steps,
            "save_steps": save_steps,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_steps": max_steps,
        }
        self._training_args = {k: v for k, v in training_args.items() if v is not None}

    def set_dataset(self, datastore: BaseDatastore, path: str):
        if os.path.isdir(path):
            if os.listdir(path):
                # if data already exists, continue
                return
        else:
            os.makedirs(path)

        # TODO: Improve this

        dataset = Dataset.from_list(
            [
                TrainerData(
                    **{
                        k: v
                        for k, v in d.items()
                        if k in TrainerData.__dataclass_fields__
                    }
                ).to_dict()
                for d in datastore.load_data()
            ]
        )
        dataset.save_to_disk(path)

    @abc.abstractmethod
    def train(
        self,
        model_id_or_path: str,
        output_dir: str,
        datastore: BaseDatastore,
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

    def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError


def make_model_dir(output_path: str):
    return os.path.join(output_path, "model")
