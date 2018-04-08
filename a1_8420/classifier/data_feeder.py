import numpy as np
import torch as tr
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

    def prototype_feeder(self,k_shot):
        matrix = self.train_data[self.data_config['train_data_len']:(self.data_config['train_data_len']+self.data_config['validation_data_len'])]
        prototypes_freq = {}
        prototypes = {}
        if k_shot == None:
            k_shot = float('inf')
        for i in range(len(matrix)):
            prototype = matrix[i][0]
            label = int(matrix[i][1])
            if label not in prototypes_freq:
                prototypes_freq[label]=1
                prototypes[label]=prototype
            else:
                if prototypes_freq[label]<k_shot:
                    prototypes_freq[label]+=1
                    prototypes[label]+=prototype
        for label in prototypes:
            prototypes[label] = (prototypes[label]/np.array(prototypes_freq[label],'float32')).astype('float32')
        prototypes_ls = []
        for i in range(10):
            prototypes_ls.append(prototypes[i])
        return prototypes_ls



    def test_feeder(self):
        dataiter = DataLoader(myDataset(self.test_data),batch_size=self.data_config['test_data_len'],shuffle='True')
        return dataiter

if __name__ == "__main__":
    data_config = {'train_data': '../dataset/optdigits.tra',
                   'test_data': '../dataset/optdigits.tes',
                   'data_filePath': '/datastore/liu121/torch_data/a1_8420/data.pkl',
                   'train_data_len': 1934,
                   'validation_data_len': 946,
                   'test_data_len': 1797,
                   'batch_size': 30
                   }
    df = DataFeeder(data_config)
    dataiter = df.train_data()
    for X,y_ in dataiter:
        print(X)
        print(y_)