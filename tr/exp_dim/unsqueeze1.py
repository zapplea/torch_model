import numpy as np
import torch as tr

v = tr.FloatTensor(np.zeros(shape=(30,),dtype='float32'))
result = tr.unsqueeze(v,dim=1)
print(result.size())