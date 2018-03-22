import numpy as np
import torch as tr
import torch.nn.functional as F
v = tr.FloatTensor(np.zeros(shape=(3,3),dtype='float32'))
result = tr.sum(v,dim=1)
print(result)