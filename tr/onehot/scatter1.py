import numpy as np
import torch as tr
import torch.nn.functional as F
y_=tr.LongTensor(np.array([[1],[4],[3],[2],[1],[2],[2],[1],[3],[0]],dtype='float32'))
result = tr.zeros(10,5).scatter_(1,y_,1)
print(result)