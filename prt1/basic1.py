import torch as tr
import numpy as np

c1 = tr.FloatTensor(np.ones(shape=(3,3),dtype='float32'))
c2 = tr.FloatTensor(np.ones(shape=(3,3)))

result = tr.add(c1,c2)
print(result)

v1 = tr.autograd.Variable(c1)