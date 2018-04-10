import numpy as np
import torch as tr
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

# TODO: need to normalize data
class myDataset(Dataset):
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels
        self.length = len(features)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        instance = (self.features[index],self.labels[index])
        return instance

class DataFeeder:
    def __init__(self,data_config):
        self.data_config = data_config
        self.train_features,self.train_labels,self.test_features,self.test_labels=self.loader()

    def loader(self):
        with open(self.data_config['data_filePath'],'rb') as f:
            data = pickle.load(f)
        return data['train_features'], data['train_labels'],data['test_features'],data['test_labels']

    def train_feeder(self):
        dataiter = DataLoader(myDataset(self.train_features[:self.data_config['train_data_len']],
                                        self.train_labels[:self.data_config['train_data_len']]),
                              batch_size=self.data_config['batch_size'],
                              shuffle='True')
        return dataiter

    def validation_feeder(self):
        dataiter = DataLoader(myDataset(self.train_features[self.data_config['train_data_len']:(self.data_config['train_data_len']+self.data_config['validation_data_len'])],
                                        self.train_labels[self.data_config['train_data_len']:(self.data_config['train_data_len']+self.data_config['validation_data_len'])]),
                              batch_size=self.data_config['validation_data_len'],
                              shuffle='True')
        return dataiter

    def prototype_feeder(self,k_shot):
        features = self.train_features[self.data_config['train_data_len']:(self.data_config['train_data_len']+self.data_config['validation_data_len'])]
        labels = self.train_labels[self.data_config['train_data_len']:(self.data_config['train_data_len']+self.data_config['validation_data_len'])]
        length = len(features)
        prototypes_freq = {}
        prototypes = {}
        if k_shot == None:
            k_shot = float('inf')
        for i in range(length):
            prototype = features[i]
            label = labels[i]
            if label not in prototypes_freq:
                prototypes_freq[label]=1
                prototypes[label]=prototype
            else:
                if prototypes_freq[label]<k_shot:
                    prototypes_freq[label]+=1
                    prototypes[label]+=prototype
        for label in prototypes:
            prototypes[label] = (prototypes[label]/np.array(prototypes_freq[label],'float32'))
        prototypes_ls = []
        for i in range(10):
            prototypes_ls.append(prototypes[i])
        return prototypes_ls



    def test_feeder(self):
        dataiter = DataLoader(myDataset(self.test_features,
                                        self.test_labels),
                              batch_size=self.data_config['test_data_len'],
                              shuffle='True')
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
    dataiter = df.train_feeder()
    for X,y_ in dataiter:
        print('feature size: ',X.size())
        print('label size: ',y_.size())