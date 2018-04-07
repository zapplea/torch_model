import torch as tr
import numpy as np


class Net(tr.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def foward(self, X):
        W = tr.autograd.Variable(tr.FloatTensor(np.random.randn(3, 4)), requires_grad=True)
        z = tr.matmul(X, W)
        return z

    def foward(self, X):
        W = tr.autograd.Variable(tr.FloatTensor(np.random.randn(3, 4)), requires_grad=True)
        z = tr.matmul(X, W)


module = Net()
print(module.parameters())
for para in list(module.parameters()):
    print(para)