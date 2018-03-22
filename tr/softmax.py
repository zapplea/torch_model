import numpy as np
import torch as tr
import torch.nn.functional as F

F.softmax(tr.FloatTensor(np.ones(shape=(3,3,3))),dim=-1)