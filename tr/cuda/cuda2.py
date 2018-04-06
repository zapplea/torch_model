import torch as tr
import torch.nn.functional as F

class Net(tr.nn.Module):
    def forward(self,X):
        result = F.softmax(tr.autograd.Variable(X,requires_grad=False),dim=1)
        return result

with tr.cuda.device(0):
    module= Net()
    X = tr.ones((3,3))
    print(X.numpy())
    y = tr.add(tr.autograd.Variable(X),tr.autograd.Variable(X))
    outputs = module.forward(X)

    print(outputs.data.numpy())