import torch as tr
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy
import sklearn.metrics as metrics

class DataGenerator:
    def __init__(self,data_config):
        self.data_config = data_config
        self.train_data, self.test_data = self.data_loader()

    def data_loader(self):
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.data_config['cuda'] else {}
        train_data = tr.utils.data.DataLoader(
            datasets.MNIST(self.data_config['data_filePath'], train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=self.data_config['batch_size'], shuffle=True, **kwargs)
        test_data = tr.utils.data.DataLoader(
            datasets.MNIST(self.data_config['data_filePath'], train=False,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=self.data_config['batch_size'], shuffle=True, **kwargs)
        return train_data, test_data

    def feed_train(self):
        return enumerate(self.train_data)

    def feed_test(self):
        return enumerate(self.test_data)


class Net(tr.nn.Module):
    def __init__(self, nn_config):
        super(Net,self).__init__()
        self.nn_config=nn_config

    def forward(self, X):
        """
        
        :param X: (batch size, feature dim)
        :return: 
        """
        X = X.view(-1,784)
        W = tr.autograd.Variable(tr.randn(1000,784))
        bias = tr.autograd.Variable(tr.zeros())
        h = F.tanh(F.linear(X,W,bias=bias))
        score = F.softmax(h,dim=-1)
        return score

    def loss(self,X,y_):
        Y = tr.zeros(self.nn_config['batch_size'],self.nn_config['labels_num']).scatter_(1,y_,1)
        score = self.forward(X)
        tr.sum(tr.mul(score,y_),dim=1)

    def prediction(self,X):
        score = self.forward(X)
        values,indices = tr.max(score)
        return indices

class Classifier():
    def __init__(self, nn_config,data_config):
        self.nn_config = nn_config
        self.data_config = data_config
        self.dg = DataGenerator(data_config)

    def classifier(self):
        if self.nn_config['cuda']:
            net = Net(self.nn_config).cuda(0)
        else:
            net =Net(self.nn_config)
        return net

    def train(self):
        graph = self.classifier()
        train_data = self.dg.feed_train()
        test_data = self.dg.feed_test()
        for batch_id,(X,y_) in train_data:
            if self.nn_config['cuda']:
                X = X.cuda()
            graph.forward(X)


if __name__ == "__main__":
    cuda = True
    batch_size = 30
    data_config = {'batch_size':batch_size,
                   'cuda':cuda,
                   'data_filePath':'/media/data2tb1/yibing/nosqldb/tr_data/MNIST'}
    nn_config = {'batch_size':batch_size,
                 'labels_num':10,
                 'cuda':cuda,
                 'epoch':1000}
    dg = DataGenerator(data_config)
    train_data,test_data = dg.data_loader()
    for batch_id, (X,Y_) in enumerate(train_data):
        # print(X.size())
        # print(X)
        X = X.view(-1,784)
        print(X.size())
        print(X)