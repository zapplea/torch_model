import torch as tr
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
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
        """
        
        :param X: 
        :param y_: (batch size,)
        :return: 
        """
        # y_.shape = (batch size, 1)
        y_=tr.unsqueeze(y_,dim=1)
        Y_ = tr.zeros(self.nn_config['batch_size'],self.nn_config['labels_num']).scatter_(1,y_,1)
        score = self.forward(X)
        loss = tr.mean(tr.sum(tr.mul(score,Y_),dim=1))
        return loss

    def prediction(self,score):
        """
        
        :param score: (batch size, number of labels)
        :return: 
        """
        pred = np.argmax(score.cpu().numpy(),axis=1)
        return pred

class Metrics():
    @staticmethod
    def accuracy(y_,y):
        acc = metrics.accuracy_score(y_,y)
        return acc

class Classifier():
    def __init__(self, nn_config,data_config):
        self.nn_config = nn_config
        self.data_config = data_config
        self.dg = DataGenerator(data_config)

    def optimizer(self,net):
        optim = tr.optim.SGD(net.parameters(),lr=self.nn_config['lr'],weight_decay=self.nn_config['weight_decay'])
        return optim

    def classifier(self):
        if self.nn_config['cuda']:
            net = Net(self.nn_config).cuda(0)
        else:
            net =Net(self.nn_config)
        return net

    def report(self,info):
        with open(self.nn_config['report'],'a+') as f:
            f.write('epoch:{}, accuracy:{}'.format(info['epoch'],info['accuracy']))

    def train(self):
        graph = self.classifier()
        if self.nn_config['cuda']:
            graph.cuda()
        train_data = self.dg.feed_train()
        test_data = self.dg.feed_test()
        optim = self.optimizer(graph)

        for epoch in range(self.nn_config['epoch']):
            for batch_id,(X,y_) in train_data:
                if self.nn_config['cuda']:
                    X = X.cuda()
                    y_= y_.cuda()
                optim.zero_grad()
                X = tr.autograd.Variable(X)
                y_ = tr.autograd.Variable(y_)
                loss = graph.loss(X,y_)
                loss.backward()
                optim.step()
            accuracy_collection = []
            for batch_id,(X,y_) in test_data:
                if self.nn_config['cuda']:
                    X = X.cuda()
                score = graph.forward(X)
                # y.type = numpy
                y = graph.prediction(score)
                accuracy_collection.append(Metrics.accuracy(y_,y))
            accuracy = np.mean(accuracy_collection)
            info = {'accuracy':accuracy,'epoch':epoch}
            self.report(info)

if __name__ == "__main__":
    cuda = True
    batch_size = 30
    data_config = {'batch_size':batch_size,
                   'cuda':cuda,
                   'data_filePath':'/media/data2tb1/yibing/nosqldb/tr_data/MNIST'}
    nn_config = {'batch_size':batch_size,
                 'labels_num':10,
                 'cuda':cuda,
                 'epoch':1000,
                 'lr':0.003,
                 'weight_decay':0.00003,
                 'report':'/'}
    dg = DataGenerator(data_config)
    train_data,test_data = dg.data_loader()
    for batch_id, (X,Y_) in enumerate(train_data):
        # print(X.size())
        # print(X)
        print(Y_.size())