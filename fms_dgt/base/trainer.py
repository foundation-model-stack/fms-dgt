# Standard
import abc


class BaseTrainer(metaclass=abc.ABCMeta):
    """This class is the base class for all trainer"""

    def __init__(self):
        pass

    @property
    @abc.abstractmethod
    def trained_model_path(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
