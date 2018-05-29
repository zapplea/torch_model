import random
import pickle
import numpy as np


class DataGenerator:
    def generator(self):
        with open('uci_optdigits/optdigits.tes') as f:
            data=[]
            for line in f:
                ls = line.split(',')
                feature = []
                for i in range(0,64):
                    feature.append(float(ls[i]))
                label = float(ls[-1])
                data.append((feature,label))
            test_features = []
            test_labels = []
            for feature, label in data:
                test_features.append(feature)
                test_labels.append(label)
            test_features = np.array(test_features,dtype='float32')
            test_labels = np.array(test_labels,dtype='int64')

        with open('uci_optdigits/optdigits.tra') as f:
            data = []
            for line in f:
                ls = line.split(',')
                feature = []
                for i in range(0,64):
                    feature.append(float(ls[i]))
                label = float(ls[-1])
                data.append((feature,label))

        train_features = []
        train_labels = []
        for feature,label in data:
            train_features.append(feature)
            train_labels.append(label)
        train_features = np.array(train_features,dtype='float32')
        train_labels = np.array(train_labels,dtype='int64')

        with open('data.pkl','wb') as f:
            pickle.dump({'test_features':test_features,
                         'test_labels':test_labels,
                         'train_features':train_features,
                         'train_labels':train_labels},f)

if __name__=="__main__":
    dg = DataGenerator()
    dg.generator()