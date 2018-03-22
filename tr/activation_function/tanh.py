import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

m = tr.FloatTensor(np.array([-1,2,0.1],dtype='float32'))
result1 = nn.Tanh()(m)
print(result1)
print('=============')
result2 = F.tanh(m)
print(result2)