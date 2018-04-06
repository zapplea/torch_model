import pickle
import numpy as np

# TODO: need to normalize data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

# TODO: need to normalize data
class myDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        instance = self.data[index]
        return instance

class DataFeeder:
    def __init__(self,data_config):
        self.data_config = data_config
        self.train_data,self.test_data=self.loader()

    def loader(self):
        with open(self.data_config['data_filePath'],'rb') as f:
            data = pickle.load(f)
        return data['train_data'], data['test_data']

    def train_feeder(self):
        dataiter = DataLoader(myDataset(self.train_data[:self.data_config['train_data_len']]),
                              batch_size=self.data_config['batch_size'],
                              shuffle='True')
        return dataiter

    def validation_feeder(self):
        dataiter = DataLoader(myDataset(self.train_data[self.data_config['train_data_len']:(self.data_config['train_data_len']+self.data_config['validation_data_len'])]),
                              batch_size=self.data_config['validation_data_len'],
                              shuffle='True')
        return dataiter

    def test_feeder(self):
        dataiter = DataLoader(myDataset(self.test_data),batch_size=self.data_config['test_data_len'],shuffle='True')
        return dataiter
# class DataFeeder:
#     def __init__(self,data_config):
#         self.data_config = data_config
#         self.train_data,self.test_data=self.loader()
#
#     def loader(self):
#         with open(self.data_config['data_filePath'],'rb') as f:
#             data = pickle.load(f)
#         return data['train_data'], data['test_data']
#
#     def data_feeder(self,mode,**kwargs):
#         if mode == 'train':
#             batch_num = kwargs['batch_id']
#             batch_size= kwargs['batch_size']
#             data_temp = self.train_data[:self.data_config['train_data_len']]
#         elif mode == 'valid':
#             data_temp = self.train_data[self.data_config['train_data_len']:(self.data_config['train_data_len']+self.data_config['validation_data_len'])]
#         else:
#             data_temp = self.test_data
#
#         if mode == 'train':
#             train_size = len(data_temp)
#             start = batch_num * batch_size % train_size
#             end = (batch_num * batch_size + batch_size) % train_size
#             if start < end:
#                 batch = data_temp[start:end]
#             elif start >= end:
#                 batch = data_temp[start:]
#                 # batch.extend(data_temp[0:end])
#         else:
#             batch = data_temp
#         X = []
#         y_ = []
#         for instance in batch:
#             X.append(instance[0])
#             y_.append(instance[1])
#
#         # during validation and test, to avoid errors are counted repeatedly,
#         # we need to avoid the same data sended back repeately
#         return (np.array(X, dtype='float32'), np.array(y_, dtype='int64'))
#
#     def test_feeder(self):
#
#         return