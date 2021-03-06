import numpy as np
import torch as tr
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle


# TODO: need to normalize data
class myDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.length = len(features)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        instance = (self.features[index], self.labels[index])
        return instance


class DataFeeder:
    def __init__(self, data_config):
        self.data_config = data_config
        self.train_features, self.train_labels, self.test_features, self.test_labels = self.loader()

    def loader(self):
        with open(self.data_config['data_filePath'], 'rb') as f:
            data = pickle.load(f)
        return data['train_features'], data['train_labels'], data['test_features'], data['test_labels']

    def query_feeder(self):
        dataiter = DataLoader(
            myDataset(self.train_features[:self.data_config['query_data_len']],
                      self.train_labels[:self.data_config['query_data_len']]),
            batch_size=self.data_config['batch_size'],
            shuffle='True')
        return dataiter

    # only for prototypical network
    def prototype_feeder(self, k_shot):
        """

        :param k_shot: 
        :return: shape = (labels number, k_shot, feature dim)
        """
        support_features = self.train_features[self.data_config['query_data_len']:(self.data_config['query_data_len']-1+self.data_config['support_data_len'])]
        support_labels = self.train_labels[self.data_config['query_data_len']:(self.data_config['query_data_len']-1+self.data_config['support_data_len'])]
        length = len(support_features)
        prototypes_freq = {}
        prototypes = {}
        if k_shot == None:
            k_shot = 50
        for i in range(length):
            prototype = support_features[i]
            label = support_labels[i]
            if label not in prototypes_freq:
                prototypes_freq[label] = 1
                prototypes[label] = [prototype]
            else:
                if prototypes_freq[label] < k_shot:
                    prototypes_freq[label] += 1
                    prototypes[label].append(prototype)
        prototypes_ls = []
        for i in range(10):
            prototypes_ls.append(prototypes[i])
        return np.array(prototypes_ls, dtype='float32')

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
    # dataiter = df.train_feeder()
    # for X,y_ in dataiter:
    #     print('feature size: ',X.size())
    #     print('label size: ',y_.size())
    prototypes = df.prototype_feeder(k_shot=40)
    for p in prototypes:
        print(type(p))
    print(type(prototypes))
    print(tr.FloatTensor(np.array(prototypes, 'float32')))