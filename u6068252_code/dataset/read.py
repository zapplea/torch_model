import pickle

def read_mnist():
    with open('data_mnist.pkl','rb') as f:
        data = pickle.load(f)
        train_instances =data['train_features']
        test_instances = data['test_features']
        test_labels = data['test_labels']
        print('len_train: ', len(train_instances))
        print('len_test: ', len(test_instances))
        # for ins in test_instances:
        #     print(ins)
        #     break
        # for label in test_labels:
        #     print(label)
        #     break

def read_opt():
    with open('data.pkl','rb') as f:
        data = pickle.load(f)
        test_instances = data['test_features']
        test_labels = data['test_labels']
        train_instances = data['train_features']

        for ins in test_instances:
            print(ins)
            break
        for label in test_labels:
            print(label)
            break


read_mnist()