import torch as tr
import torch.nn.functional as F
import numpy as np
import sklearn

class Net(tr.nn.Module):
    def __init__(self,nn_config):
        super(Net,self).__init__()
        self.nn_config = nn_config

    def foward(self,X):
        in_dim = self.nn_config['feature_dim']
        out_dim = self.nn_config['layer_dim'][0]
        W = tr.autograd.Variable(tr.randn((in_dim,out_dim)),requires_grad=True)
        linear_layer = tr.matmul(X,W)
        score = F.softmax(linear_layer,dim=1)
        return score

class Cascading:
    def __init__(self,nn_config,df):
        self.nn_config = nn_config
        self.df = df

    def optimizer(self,model):
        optim = tr.optim.SGD(model.parameters(),lr=self.nn_config['lr'],weight_decay=self.nn_config['weight_decay'])
        return optim

    def cross_entropy_loss(self,input,target):
        input = tr.log(input)
        loss = tr.nn.NLLLoss(size_average=True,reduce=True)
        return loss(input,target)

    def prediction(self,score):
        # TODO: need to use u>theta to compute which is predicted label
        score = score.data.numpy()
        condition = np.greater_equal(score,np.array(self.nn_config['theta'],dtype='float32'))
        score = np.where(condition,score,np.zeros_like(score,dtype='float32'))
        pred_labels=[]
        for i in score.shape[0]:
            instance = score[i]
            if str(np.all(np.equal(instance,np.array(0,dtype='float32')))) == 'True':
                pred_labels.append(-1)
            else:
                pred_labels.append(np.argmax(instance))
        return np.array(pred_labels,dtype='float32')

    def metrics(self,true_labels,pred_labels):
        true_labels = list(true_labels)
        pred_labels = list(pred_labels)
        true_ls = []
        pred_ls = []
        for i in range(pred_labels.shape[0]):
            if pred_labels[i] == -1:
                continue
            else:
                true_ls.append(true_labels[i])
                pred_ls.append(pred_labels[i])
        true_labels = np.array(true_ls,dtype='float32')
        pred_labels = np.array(pred_ls,dtype='float32')
        f1 = sklearn.metrics.f1_score(true_labels=true_labels,pred_labels=pred_labels, average='macro')
        accuracy = sklearn.metrics.accuracy_score(true_labels=true_labels,pred_labels=pred_labels)
        return f1,accuracy

    def classifier(self):
        model = Net(self.nn_config)
        for epoch in range(self.nn_config['epoch']):
            self.train(model)
            knn_features,knn_labels = self.validation(model)
            self.test(model,knn_features,knn_labels)


    def train(self,model):
        dataiter = self.df.train_feeder()
        optim = self.optimizer(model)
        for X,y_ in dataiter:
            optim.zero_grad()
            score = model.forward(tr.autograd.Variable(X,requires_grad=False))
            loss = self.cross_entropy_loss(score,tr.autograd.Variable(y_,requires_grad=False))
            loss.backward()
            optim.step()

    def knn_matrix_generator(self,true_lables,pred_labels,X):
        knn_features = []
        knn_labels = []
        for i in range(pred_labels.shape[0]):
            if pred_labels[i] == -1:
                knn_features.append(X[i].numpy())
                knn_labels.append(true_lables[i])
        return knn_features,knn_labels

    def validation(self,model):
        f = open(self.nn_config['report_filePath'],'w+')
        dataiter = self.df.validation_feeder()
        for X,y_ in dataiter:
            score = model.forward(self,tr.autograd.Variable(X,requires_grad=False))
            loss = self.cross_entropy_loss(score,tr.autograd.Variable(y_,requires_grad=False))
            pred_labels = self.prediction(score)
            knn_features,knn_labels = self.knn_matrix_generator(y_.numpy().astype('float32'),pred_labels,X)
            f1,accuracy = self.metrics(y_.numpy().astype('float32'),pred_labels)
            f.write('Validation: loss:{:.4f}, accuracy:{:.4f}, f1:{:.4f}'.format(loss.data.numpy(),f1,accuracy))
        f.close()
        return knn_features,knn_labels

    def knn(self,knn_features,knn_labels,X,pred_labels):
        exceptions = []
        exceptions_index = []
        for i in range(pred_labels.shape[0]):
            if pred_labels[i] == -1:
                exceptions.append(X[i])
                exceptions_index.append(i)
        neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbours=self.nn_config['neigh_num'])
        neigh.fit(knn_features,knn_labels)
        exceptions_labels = neigh.predict(exceptions)
        for i in range(len(exceptions_labels)):
            index = exceptions_index[i]
            pred_labels[index] = exceptions_labels[i]
        return pred_labels

    def test(self,model,knn_features,knn_labels):
        f=open(self.nn_config['report_filePath'],'a+')
        test_data =self.df.test_feeder()
        for X, y_ in test_data:
            score = model.forward(tr.autograd.Variable(X,requires_grad=True))
            loss = self.cross_entropy_loss(score,tr.autograd.Variable(y_,requires_grad=False))
            pred_labels = self.prediction(score)
            pred_labels = self.knn(knn_features,knn_labels,X,pred_labels)
            f1,accuracy = self.metrics(true_labels=y_.numpy().astype('float32'),pred_labels=pred_labels)
            f.write('Test: loss:{:.4f}, accuracy:{:.4f}, f1:{:.4f}'.format(loss.data.numpy(), f1, accuracy))
        f.close()