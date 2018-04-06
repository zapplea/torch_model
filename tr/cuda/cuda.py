import torch as tr
import torch.nn.functional as F

class Net(tr.nn.Module):
    def forward(self,X):
        result = F.softmax(tr.autograd.Variable(X,requires_grad=False),dim=1)
        print(result.get_device())
        return result

with tr.cuda.device(0):
    module= Net()
    module.cuda()
    X = tr.ones((3,3))
    X = X.cuda()
    print(X.get_device())
    outputs = module.forward(X)
    print(outputs.data.numpy())