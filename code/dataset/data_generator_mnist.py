import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import random

class DataGenerator:
    def __init__(self, data_config):
        self.data_config=data_config

    def data_generator_full(self):
        filename_list = []
        for (root, dirs, files) in os.walk(self.data_config['folder']):
            for file in files:
                filename_list.append(root + '/' + file)
        random.shuffle(filename_list)
        random.shuffle(filename_list)
        attributes={}
        labels={}
        for filename in filename_list:
            img = mpimg.imread(filename)
            label=int(filename.split('/')[-2])
            if 'training' in filename:
                if 'train' in attributes:
                    attributes['train'].append(img)
                    labels['train'].append(label)
                else:
                    attributes['train']=[img]
                    labels['train']=[label]
            elif 'testing' in filename:
                if 'test' in attributes:
                    attributes['test'].append(img)
                    labels['test'].append(label)
                else:
                    attributes['test']=[img]
                    labels['test']=[label]


        with open('data_mnist.pkl','wb') as f:
            pickle.dump({'test_features':attributes['test'],
                         'test_labels':labels['test'],
                         'train_features':attributes['train'],
                         'train_labels':labels['train']},f)


    def data_generator_few_shot_org(self):
        """
        In this function, we split the mnist to Dp,Dt and test
        :return: 
        """
        with open(self.data_config['data_mnist_filePath'],'rb') as f:
            data = pickle.load(f)
            train_features = data['train_features']
            train_labels = data['train_labels']

            continue_point=0

            Dt_features = []
            Dt_labels = []
            freq_Dt = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for i in range(len(train_features)):
                train_feature = train_features[i]
                train_label = train_labels[i]
                if train_label in freq_Dt and freq_Dt[train_label]<self.data_config['Dt_threshold']:
                    Dt_features.append(train_feature)
                    Dt_labels.append(train_label)
                    freq_Dt[train_label]+=1
                flag=True
                for key in freq_Dt:
                    if freq_Dt[key]<self.data_config['Dt_threshold']:
                        flag=False
                if flag:
                    continue_point=i
                    break
            print('freq_Dt:\n', freq_Dt)
            Dp_features = []
            Dp_labels = []
            freq_Dp = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            while continue_point<len(train_features):
                train_feature = train_features[continue_point]
                train_label = train_labels[continue_point]

                if train_label in freq_Dp and freq_Dp[train_label]<self.data_config['Dp_threshold']:
                    Dp_features.append(train_feature)
                    Dp_labels.append(train_label)
                    freq_Dp[train_label]+=1
                flag=True
                for key in freq_Dp:
                    if freq_Dp[key]<self.data_config['Dp_threshold']:
                        flag=False
                if flag:
                    break
                continue_point+=1
            print('freq_Dp:\n', freq_Dp)
            all_test_features = data['test_features']
            all_test_labels = data['test_labels']
            test_features = []
            test_labels = []
            freq_test = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
            for i in range(len(all_test_features)):
                test_feature = all_test_features[i]
                test_label = all_test_labels[i]
                if test_label in freq_test and freq_test[test_label]<self.data_config['test_threshold']:
                    test_features.append(test_feature)
                    test_labels.append(test_label)
                    freq_test[test_label]+=1
                flag=True
                for key in freq_test:
                    if freq_test[key]<self.data_config['test_threshold']:
                        flag=False
                if flag:
                    break
            print('freq_test:\n', freq_test)
            with open(self.data_config['data_mnist_cnn_filePath'], 'wb') as f:
                pickle.dump({'Dt_features': Dt_features,
                             'Dt_labels': Dt_labels,
                             'Dp_features': Dp_features,
                             'Dp_labels':Dp_labels,
                             'test_features':test_features,
                             'test_labels':test_labels}, f)

        return continue_point

    def data_generator_few_shot_leg(self,continue_point):
        """
        In this function, we split the mnist to Dp,Dt,Du and test; Du doesn't include new labels
        :return: 
        """
        with open(self.data_config['data_mnist_cnn_filePath'],'rb') as f:
            data = pickle.load(f)
            with open(self.data_config['data_mnist_filePath'], 'rb') as f:
                mnist_data = pickle.load(f)
                train_features = mnist_data['train_features']
                train_labels = mnist_data['train_labels']

                Du_features = []
                Du_labels = []
                freq_Du = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
                while continue_point < len(train_features):
                    train_feature = train_features[continue_point]
                    train_label = train_labels[continue_point]

                    if train_label in freq_Du and freq_Du[train_label] < self.data_config['Du_threshold']:
                        Du_features.append(train_feature)
                        Du_labels.append(train_label)
                        freq_Du[train_label] += 1
                    flag = True
                    for key in freq_Du:
                        if freq_Du[key] < self.data_config['Du_threshold']:
                            flag = False
                    if flag:
                        break
                    continue_point += 1
                print('freq_Du_leg:\n', freq_Du)
                data['Du_features'] = Du_features
                data['Du_labels'] = Du_labels
                with open(self.data_config['data_mnist_leg_filePath'], 'wb') as f:
                    pickle.dump(data, f)

    def data_generator_few_shot_illeg(self):
        """
        In this function, we split the mnist to Dp,Dt,Du and test; Du include new labels
        :return: 
        """
        with open(self.data_config['data_mnist_leg_filePath'],'rb') as f:
            data = pickle.load(f)
            Du_features = data['Du_features']
            Du_labels = data['Du_labels']
            with open(self.data_config['data_mnist_filePath'], 'rb') as f:
                mnist_data = pickle.load(f)
                train_features = mnist_data['train_features']
                train_labels = mnist_data['train_labels']

                freq_Du = {5: 0, 6: 0, 7: 0}
                for i in range(len(train_features)):
                    train_feature = train_features[i]
                    train_label = train_labels[i]
                    if train_label in freq_Du and freq_Du[train_label] < self.data_config['Du_threshold']:
                        Du_features.append(train_feature)
                        Du_labels.append(train_label)
                        freq_Du[train_label] += 1
                    flag = True
                    for key in freq_Du:
                        if freq_Du[key] < self.data_config['Du_threshold']:
                            flag = False
                    if flag:
                        break
            print('freq_Du_illeg:\n',freq_Du)
            data['Du_features'] = Du_features
            data['Du_labels'] = Du_labels
            with open(self.data_config['data_mnist_illeg_filePath'], 'wb') as f:
                pickle.dump(data, f)

if __name__=="__main__":
    data_config={'folder':'/home/yibing/Documents/course2018_1/dataset/mnist_png',
                 'data_mnist_filePath':'data_mnist.pkl',
                 'data_mnist_cnn_filePath':'data_mnist_cnn.pkl',
                 'data_mnist_leg_filePath':'data_mnist_leg.pkl',
                 'data_mnist_illeg_filePath':'data_mnist_illeg.pkl',
                 'Dt_threshold':10,
                 'Dp_threshold':5,
                 'Du_threshold':5,
                 'test_threshold':50}
    dg = DataGenerator(data_config)
    dg.data_generator_full()
    continue_point = dg.data_generator_few_shot_org()
    dg.data_generator_few_shot_leg(continue_point)
    dg.data_generator_few_shot_illeg()