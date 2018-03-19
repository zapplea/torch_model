import torch as tr
import torch.nn.functional as F
from torchvision import datasets, transforms

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
        print(len(data))
        count = 0
        for id, (instance,label) in data:
            print(id)
            print(instance)
            print(label)
            count+=1
            if count == 1:
                break
        # for batch_id, (data,target) in enumerate(self.train_data):
        #     print(data)


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
                   'data_filePath':'/datastore/liu121/torch_data'}
    dg = DataGenerator(data_config)
    dg.feed_train()