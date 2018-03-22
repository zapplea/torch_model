import numpy as np
import torch as tr

result = tr.mul(tr.autograd.Variable(tr.FloatTensor(np.ones(shape=(3,3),dtype='float32'))),
                tr.autograd.Variable(tr.FloatTensor(np.ones(shape=(3,3),dtype='float32')*3)))
print(result.data)

a = tr.autograd.Variable(tr.FloatTensor(np.ones(shape=(3,3),dtype='float32')))
print(a)