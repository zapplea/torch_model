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
        self.data = self.loader()

    def loader(self):
        with open(self.data_config['data_filePath'], 'rb') as f:
            data = pickle.load(f)

        return data


    def query_feeder(self):
        dataiter = DataLoader(
            myDataset(self.data['Dt_features'],
                      self.data['Dt_labels']),
            batch_size=self.data_config['batch_size'],
            shuffle='True')
        return dataiter

    # only for prototypical network
    def prototype_feeder(self):
        """

        :param k_shot: 
        :return: shape = (labels number, k_shot, feature dim)
        """
        support_features = self.data['Dp_features']
        support_labels = self.data['Dp_labels']
        dic={}
        for i in range(len(support_features)):
            feature=support_features[i]
            label=support_labels[i]
            if label in dic:
                dic[label].append(feature)
            else:
                dic[label]=[feature]

        prototypes_ls = []
        for i in range(5):
            prototypes_ls.append(dic[i])
        return np.array(prototypes_ls, dtype='float32')

    def unlabeled_feeder(self):
        return self.data['Du_features']

    def test_feeder(self):
        dataiter = DataLoader(myDataset(self.data['test_features'],
                                        self.data['test_labels']),
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