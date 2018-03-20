import torch as tr
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy

class DataGenerator:
    def __init__(self,data_config):
        self.data_config = data_config
        self.train_data, self.test_data = self.data_loader()

    def data_loader(self):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.data_config['cuda'] else {}
        train_data = tr.utils.data.DataLoader(
            datasets.MNIST(self.data_config['data_filePath'], train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.data_config['batch_size'], shuffle=True, **kwargs)
        test_data = tr.utils.data.DataLoader(
            datasets.MNIST(self.data_config['data_filePath'], train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.data_config['batch_size'], shuffle=True, **kwargs)
        return train_data, test_data

    def feed_train(self):
        data = enumerate(self.train_data)
        return data
        # for id, (instances,labels) in data:
        #     return instances, labels

    def feed_test(self):
        data = enumerate(self.test_data)


class Net(tr.nn.Module):
    def __init__(self, nn_config):
        super(Net,self).__init__()
        self.nn_config=nn_config

    def forward(self, x):
        pass

class Classifier():
    def __init__(self, nn_config,data_config):
        self.nn_config=nn_config

    def classifier(self):
        pass

    def train(self):
        pass


if __name__ == "__main__":
    data_config = {'batch_size':30,'cuda':True,
                   'data_filePath':'/media/data2tb1/yibing/nosqldb/tr_data/MNIST'}
    dg = DataGenerator(data_config)
    data = dg.feed_train()
    print(type(data))