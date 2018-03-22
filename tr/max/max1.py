import torch as tr
import torch.nn.functional as F
import numpy as np

arr = np.random.uniform(size=(10,)).astype('float32')
result = tr.max(tr.FloatTensor(arr))
print(result)