import torch as tr
import torch.nn.functional as F
import numpy as np
import sklearn.metrics
import sklearn.neighbors

class Net(tr.nn.Module):
    """
    computation graph of Cascading Network
    """
    def __init__(self,nn_config):
        super(Net,self).__init__()
        self.nn_config = nn_config
        in_dim = self.nn_config['feature_dim']
        out_dim = self.nn_config['label_dim']
        self.linear = tr.nn.Linear(in_dim,out_dim,bias=True)

    def forward(self,X):
        """
        parametric linear model in cascading network used to learn ruels. 
        input image to linear layer and then use softmax to calculate distribute of the image over each class.
        :param X: 
        :return: 
        """
        # linear_layer = self.linear(X)
        linear_layer = self.linear(X)
        score = F.softmax(linear_layer,dim=1)
        return score

    def cross_entropy_loss(self,input,target):
        """
        calculate loss for result of softmax.
        :param input: 
        :param target: 
        :return: 
        """
        input = tr.log(input)
        loss = tr.nn.NLLLoss(size_average=True,reduce=True)
        return loss(input,target)

class Cascading:
    """
    The cascading classifier learn a parametric linear model and a non-parametric K-NN. first use training dataset to train the linear 
    model. Then use support dataset (or validation dataset) to test the linear model. If probability of the image over each class is smaller 
    than a theta (here equal to 0.6), then the image will be rejected by the linear model and given to K-NN. The K-NN will use exceptions whose 
    classes are know as support points to classify a query image. During test, we use writer independent model to test it. First the image 
    in test will be input into linear model to predict its class. Then if the linear model reject the test image, we use K-NN to classify.
    
    """
    def __init__(self,nn_config,df):
        self.nn_config = nn_config
        self.df = df

    def optimizer(self,module):
        """
        optimizer
        :param module: 
        :return: 
        """
        optim = tr.optim.SGD(module.parameters(),lr=self.nn_config['lr'],weight_decay=self.nn_config['weight_decay'])
        return optim

    def prediction(self,score):
        """
        predict to which class the image belongs for linear model in cascading network.
        :param score: 
        :return: 
        """
        condition = np.greater_equal(score,np.array(self.nn_config['theta'],dtype='float32'))
        score = np.where(condition,score,np.zeros_like(score,dtype='float32'))
        pred_labels=[]
        for i in range(len(score)):
            instance = score[i]
            if str(np.all(np.equal(instance,np.array(0,dtype='float32')))) == 'True':
                pred_labels.append(-1)
            else:
                pred_labels.append(np.argmax(instance))
        return np.array(pred_labels,dtype='float32')

    def metrics(self,true_labels,pred_labels):
        """
        calculate f1 score and accuracy to evaluate the networks' performance
        :param true_labels: 
        :param pred_labels: 
        :return: 
        """
        true_labels = list(true_labels)
        pred_labels = list(pred_labels)
        true_ls = []
        pred_ls = []
        for i in range(len(pred_labels)):
            if pred_labels[i] == -1:
                continue
            else:
                true_ls.append(true_labels[i])
                pred_ls.append(pred_labels[i])
        true_labels = np.array(true_ls,dtype='float32')
        pred_labels = np.array(pred_ls,dtype='float32')
        f1 = sklearn.metrics.f1_score(y_true=true_labels,y_pred=pred_labels, average='macro')
        accuracy = sklearn.metrics.accuracy_score(y_true=true_labels,y_pred=pred_labels)
        return f1,accuracy

    def classifier(self):
        """
        run the cascading network to perform classification.
        :return: 
        """
        module = Net(self.nn_config)
        if self.nn_config['cuda'] and tr.cuda.is_available():
            module.cuda()
        for epoch in range(self.nn_config['epoch']):
            with open(self.nn_config['report_filePath'],'a+') as f:
                f.write('\nepoch:{}\n'.format(epoch))
            self.train(module)
            knn_features,knn_labels = self.validation(module)
            self.test(module,knn_features,knn_labels)


    def train(self,module):
        """
        Train the cascading network's linear model.
        :param module: 
        :return: 
        """
        dataiter = self.df.train_feeder()
        optim = self.optimizer(module)
        for X,y_ in dataiter:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X,y_ = X.cuda(),y_.cuda()
            optim.zero_grad()
            score = module.forward(tr.autograd.Variable(X,requires_grad=False))
            # TODO: the size of y_ is (30,1) should be (30,)
            loss = module.cross_entropy_loss(score,tr.autograd.Variable(y_.long(),requires_grad=False))
            loss.backward()
            optim.step()

    def knn_matrix_generator(self,true_lables,pred_labels,X):
        """
        accept exceptions rejected by linear model.
        :param true_lables: 
        :param pred_labels: 
        :param X: 
        :return: 
        """
        knn_features = []
        knn_labels = []
        for i in range(pred_labels.shape[0]):
            if pred_labels[i] == -1:
                knn_features.append(X[i])
                knn_labels.append(true_lables[i])
        return knn_features,knn_labels

    def validation(self,module):
        """
        The function uses support examples to test the the linear model. If a example is rejected by the linear model when its probability
        over each class is smaller than 0.6, it will be given K-NN as exceptions. 
        :param module: 
        :return: 
        """
        dataiter = self.df.validation_feeder()
        for X,y_ in dataiter:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X,y_ = X.cuda(),y_.cuda()
            score = module.forward(tr.autograd.Variable(X,requires_grad=False))
            loss = module.cross_entropy_loss(score,tr.autograd.Variable(y_.long(),requires_grad=False))
            if self.nn_config['cuda'] and tr.cuda.is_available():
                score = score.cpu()
                y_= y_.cpu()
                X = X.cpu()
                loss = loss.cpu()
            pred_labels = self.prediction(score.data.numpy())
            knn_features, knn_labels = self.knn_matrix_generator(y_.numpy().astype('float32'), pred_labels, X.numpy())
            f1, accuracy = self.metrics(y_.numpy().astype('float32'), pred_labels)
        return knn_features,knn_labels

    def knn(self,knn_features,knn_labels,X,pred_labels):
        """
        A K nearest neighbor algorithm that classifiy exceptions reject by linear model when test with writer independent images.
        :param knn_features: 
        :param knn_labels: 
        :param X: 
        :param pred_labels: 
        :return: 
        """
        exceptions = []
        exceptions_index = []
        for i in range(pred_labels.shape[0]):
            if pred_labels[i] == -1:
                exceptions.append(X[i])
                exceptions_index.append(i)
        neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=self.nn_config['neigh_num'])
        neigh.fit(knn_features,knn_labels)
        exceptions_labels = neigh.predict(exceptions)
        for i in range(len(exceptions_labels)):
            index = exceptions_index[i]
            pred_labels[index] = exceptions_labels[i]
        return pred_labels

    def test(self,module,knn_features,knn_labels):
        """
        Test the cascadingng classifier
        :param module: 
        :param knn_features: 
        :param knn_labels: 
        :return: 
        """
        f=open(self.nn_config['report_filePath'],'a+')
        test_data =self.df.test_feeder()
        for X, y_ in test_data:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X,y_ = X.cuda(),y_.cuda()
            score = module.forward(tr.autograd.Variable(X,requires_grad=True))
            loss = module.cross_entropy_loss(score,tr.autograd.Variable(y_.long(),requires_grad=False))

            if self.nn_config['cuda'] and tr.cuda.is_available():
                score = score.cpu()
                y_= y_.cpu()
                X = X.cpu()
                loss = loss.cpu()

            pred_labels = self.prediction(score.data.numpy())
            pred_labels = self.knn(knn_features,knn_labels,X.numpy(),pred_labels)
            f1,accuracy = self.metrics(true_labels=y_.numpy().astype('float32'),pred_labels=pred_labels)
            f.write('Test: loss:{:.4f}, accuracy:{:.4f}, f1:{:.4f}\n'.format(float(loss.data.numpy()), float(f1), float(accuracy)))
            f.flush()
        f.close()