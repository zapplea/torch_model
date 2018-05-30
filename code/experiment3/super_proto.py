import torch as tr
import torch.nn.functional as F
import numpy as np
import sklearn.metrics

class Net(tr.nn.Module):
    """
    computation graph of Prototypical Network
    """
    def __init__(self,nn_config,**kwargs):
        super(Net,self).__init__()
        self.nn_config = nn_config

        # cnn1: 28x28 --> 14x14
        self.conv1=tr.nn.Sequential(
            tr.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2),
            tr.nn.ReLU(),
            tr.nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # cnn2: 14x14 --> 7x7
        self.conv2=tr.nn.Sequential(
            tr.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            tr.nn.ReLU(),
            tr.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # linear: 7x7x64
        in_dim=self.nn_config['cnn_feature_dim']
        out_dim=self.nn_config['connect_layer_dim']
        self.linear1 = tr.nn.Linear(in_dim,out_dim, bias=True)
        if self.nn_config['is_share_weight']:
            self.linear1.weight=tr.nn.Parameter(kwargs['weight_initial'],requires_grad=True)
            self.linear1.bias = tr.nn.Parameter(kwargs['bias_initial'], requires_grad=True)

    def forward_cnn(self,X):
        cnn_layer1 = self.conv1(X)
        cnn_layer2 = self.conv2(cnn_layer1)
        cnn_layer2 = cnn_layer2.view(-1,self.nn_config['cnn_feature_dim'])
        return cnn_layer2

    def forward_nonlinear(self,X):
        """
        The nonlinear function. It maps embedding of images to anonther embedding space
        :param X: 
        :return: 
        """
        linear_layer1 = self.linear1(X)
        hidden_layer = F.tanh(linear_layer1)
        return hidden_layer

    def forward_query(self,X):
        """
        mapping the query image to another embedding space.
        :param X: 
        :return: 
        """
        conv = self.forward_cnn(X)
        return self.forward_nonlinear(conv)

    def forward_prot(self,C):
        """
        mapping support images to another embedding space and  calculate prototype for each class.
        :param C: shape = (labels number, k_shot, feature dim)
        :return: 
        """
        C = C.view(-1,1,self.nn_config['feature_height_dim'],self.nn_config['feature_width_dim'],1)
        C = self.forward_cnn(C)
        C = self.forward_nonlinear(C)
        C = C.view(self.nn_config['label_dim'],self.nn_config['k_shot'],self.nn_config['connect_layer_dim'])
        # shape = (labels num, feature dim)
        C = C.mean(1)
        return C

    def forward_softmax(self,X,C):
        """
        It calculate to which class each image belongs.
        :param X: query image
        :param C: prototypes of all classes
        :return: 
        """
        X = self.forward_query(X)
        # shape = (batch size, 1, hidden layer dim)
        X = tr.unsqueeze(X,dim=1)
        # shape = (batch size, labels num, fully connected layer dim)
        X = X.repeat(1,self.nn_config['label_dim'],1)
        # shape = (labels num, fully connected layer dim)
        C = self.forward_prot(C)
        # shape = (batch size, labels num)
        euclidean_distance = tr.sqrt(tr.mul(tr.add(X,-C),tr.add(X,-C)).sum(2))
        score = F.softmax(-euclidean_distance,dim=1)
        return score

    def cross_entropy_loss(self,input,target):
        """
        loss function
        :param input: 
        :param target: 
        :return: 
        """
        input = tr.log(input)
        loss = tr.nn.NLLLoss(size_average=True,reduce=True)
        return loss(input,target)

class ImgCompNet(tr.nn.Module):
    """
    computation graph of Shared Weights Network
    """
    def __init__(self,nn_config):
        super(ImgCompNet,self).__init__()
        self.nn_config = nn_config
        in_dim = self.nn_config['feature_height_dim']*self.nn_config['feature_width_dim']
        out_dim = self.nn_config['connect_feature_dim']
        self.linear1 = tr.nn.Linear(in_dim,out_dim, bias=True)
        self.linear2 = tr.nn.Linear(out_dim,in_dim, bias=True)


    def compress_img(self, X):
        """
        compress image and decode it.
        :param X: 
        :return: 
        """
        X = X.view(-1,self.nn_config['feature_height_dim']*self.nn_config['feature_width_dim'])
        self.weight_average()
        hidden_layer = F.tanh(self.linear1(X))
        hidden_layer = self.linear2(hidden_layer)
        return hidden_layer

    def weight_average(self):
        """
        Average value of weights in linear1 and linear2 to let them share weights
        :return: 
        """
        average = tr.div(tr.add(self.linear1.weight.data,self.linear2.weight.t().data),2)
        self.linear1.weight.data=average
        self.linear2.weight.data=average.t()


    def MSE_loss(self, input, target):
        """
        Calcualte difference between original image and the predicted image. This is loss function
        :param X: shape=(batch size, feature dim)
        :param de_X: 
        :return: 
        """
        loss = tr.nn.MSELoss(size_average=True,reduce=True)
        return loss(input,target)


class SuperPrototypicalNet:
    def __init__(self,nn_config,df):
        self.nn_config = nn_config
        self.df = df

    def optimizer(self,model):
        """
        optimizer
        :param model: 
        :return: 
        """
        optim = tr.optim.SGD(model.parameters(),lr=self.nn_config['lr'],weight_decay=self.nn_config['weight_decay'])
        return optim

    def classifier(self):
        """
        run the prototypical network to perform classification
        :return: 
        """
        if self.nn_config['is_share_weight']:
            # train the shared-weight network
            module=ImgCompNet(self.nn_config)
            if self.nn_config['cuda'] and tr.cuda.is_available():
                module.cuda()
            for i in range(self.nn_config['comp_epoch']):
                self.train_compress(module)
            # create prototypical network
            module = Net(self.nn_config, weight_initial=module.linear1.weight.cpu().data,bias_initial=module.linear1.bias.cpu().data)
        else:
            # create prototypical network
            module = Net(self.nn_config)

        # train the prototypical network

        if self.nn_config['cuda'] and tr.cuda.is_available():
            module.cuda()
        for i in range(self.nn_config['epoch']):
            with open(self.nn_config['report_filePath'],'a+') as f:
                f.write('epoch:{}\n'.format(i))
            self.train_proto(module)
            self.test_proto(module)


    def train_compress(self,module):
        """
        Train shared weights network
        :param module: 
        :return: 
        """
        dataiter = self.df.query_feeder()
        optim = self.optimizer(module)
        for X, _ in dataiter:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X= X.cuda()
            optim.zero_grad()
            de_X = module.compress_img(tr.autograd.Variable(X,requires_grad=False))
            loss = module.MSE_loss(input = de_X,target=tr.autograd.Variable(X,requires_grad=False))
            loss.backward()
            optim.step()

    def train_proto(self,module):
        """
        train prototypical network
        :param module: 
        :return: 
        """
        dataiter = self.df.query_feeder()
        C = tr.FloatTensor(self.df.prototype_feeder())
        optim = self.optimizer(module)
        for X,y_ in dataiter:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X,y_,C = X.cuda(),y_.cuda(),C.cuda()
            X = tr.unsqueeze(X, dim=1)
            C = tr.unsqueeze(C, dim=2)
            optim.zero_grad()
            score = module.forward_softmax(tr.autograd.Variable(X,requires_grad=False),
                                           tr.autograd.Variable(C,requires_grad=False))
            loss = module.cross_entropy_loss(score,tr.autograd.Variable(y_,requires_grad=False))
            loss.backward()
            optim.step()


    def prediction(self,score):
        """
        predict to which class the image belongs based on its score
        :param score: 
        :return: 
        """
        pred_labels = np.argmax(score,axis=1).astype('float32')
        return pred_labels

    def metrics(self,true_labels,pred_labels):
        """
        calculate f1 score and accuracy to evaluate the networks' performance
        :param true_labels: 
        :param pred_labels: 
        :return: 
        """
        true_labels =list(true_labels)
        pred_labels = list(pred_labels)
        f1 = sklearn.metrics.f1_score(y_true=true_labels,y_pred=pred_labels, average='macro')
        accuracy = sklearn.metrics.accuracy_score(y_true=true_labels,y_pred=pred_labels)
        return f1,accuracy

    def test_proto(self,module):
        """
        test the prototyical network
        :param module: 
        :return: 
        """
        f=open(self.nn_config['report_filePath'],'a+')
        test_data =self.df.test_feeder()
        C = tr.FloatTensor(self.df.prototype_feeder(self.nn_config['k_shot']))

        for X, y_ in test_data:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X,y_,C = X.cuda(),y_.cuda(),C.cuda()
            X = tr.unsqueeze(X, dim=1)
            C = tr.unsqueeze(C, dim=2)
            score = module.forward_softmax(tr.autograd.Variable(X, requires_grad=False),
                                           tr.autograd.Variable(C, requires_grad=False))
            loss = module.cross_entropy_loss(score, tr.autograd.Variable(y_.long(), requires_grad=False))

            if self.nn_config['cuda'] and tr.cuda.is_available():
                score = score.cpu()
                y_= y_.cpu()
                loss = loss.cpu()

            pred_labels = self.prediction(score.data.numpy())
            f1,accuracy = self.metrics(true_labels=y_.numpy().astype('float32'),pred_labels=pred_labels)
            f.write('Test: loss:{:.4f}, accuracy:{:.4f}, f1:{:.4f}\n'.format(float(loss.data.numpy()), float(f1), float(accuracy)))
            f.flush()
        f.close()