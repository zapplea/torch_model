import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

class DataGenerator:
    def __init__(self,data_config):
        self.data_config = data_config

    def data_loader(self):
        pass

class Classifier:
    def __init__(self, nn_config):
        self.nn_config=nn_config
