from core.networks import *


class Network(BaseModel):

    def __init__(self, config):
        BaseModel.__init__(self, config)
        # Model architecture
        # Todo: read config json and build model
        # The config json might be similar the params used in hyper parameters
        # search, but the config json must have additional configuration
        # parameters
        pass

    def build_model(self):
        raise NotImplementedError

