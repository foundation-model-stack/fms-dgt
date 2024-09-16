###
# Trainer itself
###


# Standard
from typing import Any, List, Optional
import abc

# Local
from fms_dgt.base.block import DATASET_TYPE, BaseBlock


class BaseTrainerBlock(BaseBlock):
    def __init__(self, config_path: str, **kwargs: Any) -> None:
        """Initialize a block that trains a model on a dataset input.

        Parameters:
            config_path (Any): path to config used for trainer
            kwargs (Any): Additional keyword arguments to pass to the base class.
        """
        super().__init__(**kwargs)

        if len(self.arg_fields) > 1:
            raise ValueError(f"Can only accept one argument field in 'arg_fields' list")

        self._config_path = config_path

    @property
    @abc.abstractmethod
    def trained_model_path(self):
        raise NotImplementedError

    def generate(
        self,
        inputs: DATASET_TYPE,
        *,
        arg_fields: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """The generate function is the primary interface to a Block. For trainer blocks, it accepts an input dataset and trains a model.
        It returns 'None' to the user, with the trained model available at 'trained_model_path'

        Args:
            inputs (DATASET_TYPE): The dataset to train the model on

        Kwargs:
            arg_fields (Optional[List[str]]): The single field of the dataset to extract.
            **kwargs: Additional keyword args that may be passed to override the trainer parameters
        """

        if len(self.arg_fields) > 1:
            raise ValueError(f"Can only accept one argument field in 'arg_fields' list")

        dataset = []
        for x in inputs:
            inp_args, _ = self.get_args_kwargs(x, arg_fields)
            dataset.append(inp_args[0])

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
