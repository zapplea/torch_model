import torch as tr
import torch.nn.functional as F
import numpy as np

X = tr.autograd.Variable(tr.randn(3,5))
H = F.linear(X,tr.autograd.Variable(tr.randn(3,5)))
y_= tr.autograd.Variable(tr.LongTensor([1,2,0]))
output = F.nll_loss(F.log_softmax(H),y_)
print(output)
output.backward()