import torch as tr
import numpy as np


class Net(tr.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = tr.nn.Linear(3, 4)
        self.W1 = tr.nn.Parameter(tr.ones(3, 4), requires_grad=True)
        self.W2 = tr.nn.Parameter(tr.ones(3, 4))
        self.W3 = tr.add(self.W1, self.W2)

    def forward(self):
        X = tr.autograd.Variable(tr.FloatTensor(np.ones(shape=(2, 3))))
        W = self.W3
        # print(self.linear.weight)
        return W


with tr.cuda.device(0):
    model = Net()
    W = model()
    print(W.data)