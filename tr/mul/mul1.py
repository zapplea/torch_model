import numpy as np
import torch as tr

result = tr.mul(tr.FloatTensor(np.ones(shape=(3,3),dtype='float32')),
                tr.FloatTensor(np.ones(shape=(3,3),dtype='float32')*3))
print(result)