import torch as tr
import torch.nn.functional as F

class Net(tr.nn.Module):
    def forward(self,X):
        result = F.softmax(tr.autograd.Variable(X,requires_grad=False),dim=1)
        return result

module= Net()
module.cuda(device=0)
X = tr.ones((3,3))
X.cuda(0)
outputs = module.forward(X)
print(outputs.cpu().data.numpy())