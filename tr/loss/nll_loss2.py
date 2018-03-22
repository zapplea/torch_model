import torch as tr
import torch.nn.functional as F
import numpy as np


input = tr.autograd.Variable(tr.randn(3, 5),requires_grad=True)
target = tr.autograd.Variable(tr.LongTensor([1, 0, 4]))
output = F.nll_loss(F.log_softmax(input), target)
output.backward()
