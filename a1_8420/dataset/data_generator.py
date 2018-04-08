import random
import pickle
import numpy as np


class DataGenerator:
    def generator(self):
        with open('optdigits.tes') as f:
            test_data=[]
            for line in f:
                ls = line.split(',')
                feature = []
                for i in range(0,64):
                    feature.append(float(ls[i]))
                label = float(ls[-1])
                test_data.append((np.array(feature,'float32'),np.array(label,'int64')))

        with open('optdigits.tra') as f:
            train_data = []
            for line in f:
                ls = line.split(',')
                feature = []
                for i in range(0,64):
                    feature.append(float(ls[i]))
                label = float(ls[-1])
                train_data.append((np.array(feature,'float32'),np.array([label],'float32')))
        with open('/datastore/liu121/torch_data/a1_8420/data.pkl','wb') as f:
            random.shuffle(test_data)
            random.shuffle(train_data)
            pickle.dump({'test_data':test_data,'train_data':train_data},f)

if __name__=="__main__":
    dg = DataGenerator()
    dg.generator()