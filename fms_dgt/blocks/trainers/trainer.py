###
# Trainer itself
###


# Standard
from typing import Any
import abc
import os

# Third Party
from datasets import Dataset, load_from_disk

# Local
from fms_dgt.base.block import BaseBlock
from fms_dgt.base.datastore import BaseDatastore


class BaseTrainerBlock(BaseBlock):
    def __init__(self, config_path: str, **kwargs: Any) -> None:
        """Initialize a trainer that trains a model on a dataset input.

        Args:
            config_path (Any): path to config used for trainer
            kwargs (Any): Additional keyword arguments to pass to the base class.
        """
        super().__init__(**kwargs)
        self._config_path = config_path

    def get_dataset(self, datastore: BaseDatastore, path: str):
        if os.path.isdir(path):
            if os.listdir(path):
                # if data already exists, continue
                return
        else:
            os.makedirs(path)

        # TODO: Improve this
        dataset = Dataset.from_list(datastore.load_data())
        dataset.save_to_disk(path)

        return load_from_disk(path).with_format("torch")

    @abc.abstractmethod
    def train(
        self,
        model_id_or_path: str,
        output_dir: str,
        datastore: BaseDatastore,
        restart: bool = False,
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

        Kwargs:
            restart (bool): Whether to restart training or not

        Returns:
            str: Path to model that was trained
        """
        raise NotImplementedError
