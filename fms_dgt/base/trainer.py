class BaseTrainer:
    def __init__(self):
        pass

    def train(self):
        raise NotImplementedError

    def init_model(self):
        raise NotImplementedError

    def release_model(self):
        raise NotImplementedError
