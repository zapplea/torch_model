import torch as tr
import torch.nn.functional as F
import numpy as np
import sklearn.metrics

class Net(tr.nn.Module):
    def __init__(self,nn_config,**kwargs):
        super(Net,self).__init__()
        self.nn_config = nn_config
        in_dim = self.nn_config['feature_dim']
        out_dim = self.nn_config['layer_dim'][0]
        self.linear1 = tr.nn.Linear(in_dim,out_dim, bias=True)
        if self.nn_config['is_share_weight']:
            self.linear1.weight=tr.nn.Parameter(kwargs['weight_initial'],requires_grad=True)
            self.linear1.bias = tr.nn.Parameter(kwargs['bias_initial'], requires_grad=True)

    def forward_nonlinear(self,X):
        linear_layer1 = self.linear1(X)
        hidden_layer = F.tanh(linear_layer1)
        return hidden_layer

    def forward_sample(self,X):
        return self.forward_nonlinear(X)

    def forward_prot(self,C):
        """
        
        :param C: shape = (labels number, k_shot, feature dim)
        :return: 
        """
        C = C.view(-1,self.nn_config['feature_dim'])
        C = self.forward_nonlinear(C)
        C = C.view(self.nn_config['label_dim'],self.nn_config['k_shot'],self.nn_config['layer_dim'][0])
        # shape = (labels num, feature dim)
        C = C.mean(1)
        return C

    def forward_softmax(self,X,C):

        X = self.forward_sample(X)
        # shape = (batch size, 1, hidden layer dim)
        X = tr.unsqueeze(X,dim=1)
        # shape = (batch size, labels num, hidden layer dim)
        X = X.repeat(1,self.nn_config['label_dim'],1)
        C = self.forward_prot(C)
        # shape = (batch size, labels num)
        euclidean_distance = tr.sqrt(tr.mul(tr.add(X,-C),tr.add(X,-C)).sum(2))
        score = F.softmax(-euclidean_distance,dim=1)
        return score

    def cross_entropy_loss(self,input,target):
        input = tr.log(input)
        loss = tr.nn.NLLLoss(size_average=True,reduce=True)
        return loss(input,target)

class ImgCompNet(tr.nn.Module):
    def __init__(self,nn_config):
        super(ImgCompNet,self).__init__()
        self.nn_config = nn_config
        in_dim = self.nn_config['feature_dim']
        out_dim = self.nn_config['layer_dim'][0]
        self.linear1 = tr.nn.Linear(in_dim,out_dim, bias=True)
        self.linear2 = tr.nn.Linear(out_dim,in_dim, bias=True)


    def compress_img(self, X):
        self.weight_average()
        hidden_layer = F.tanh(self.linear1(X))
        hidden_layer = self.linear2(hidden_layer)
        return hidden_layer

    def weight_average(self):
        average = tr.div(tr.add(self.linear1.weight.data,self.linear2.weight.t().data),2)
        self.linear1.weight.data=average
        self.linear2.weight.data=average.t()


    def MSE_loss(self, input, target):
        """
        :param X: shape=(batch size, feature dim)
        :param de_X: 
        :return: 
        """
        loss = tr.nn.MSELoss(size_average=True,reduce=True)
        return loss(input,target)


class PrototypicalNet:
    def __init__(self,nn_config,df):
        self.nn_config = nn_config
        self.df = df

    def optimizer(self,model):
        optim = tr.optim.SGD(model.parameters(),lr=self.nn_config['lr'],weight_decay=self.nn_config['weight_decay'])
        return optim

    def classifier(self):
        if self.nn_config['is_share_weight']:
            # train the shared-weight network
            module=ImgCompNet(self.nn_config)
            if self.nn_config['cuda'] and tr.cuda.is_available():
                module.cuda()
            for i in range(self.nn_config['comp_epoch']):
                # with open(self.nn_config['report_filePath'],'a+') as f:
                #     f.write('ImgCompNet_epoch:{}\n'.format(i))
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
                f.write('ProtoNet_epoch:{}\n'.format(i))
            self.train_proto(module)
            self.test(module)


    def train_compress(self,module):
        dataiter = self.df.train_feeder()
        optim = self.optimizer(module)
        for X, _ in dataiter:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X= X.cuda()
            optim.zero_grad()
            de_X = module.compress_img(tr.autograd.Variable(X,requires_grad=False))
            loss = module.MSE_loss(input = de_X,target=tr.autograd.Variable(X,requires_grad=False))
            loss.backward()
            optim.step()
        # with open(self.nn_config['report_filePath'], 'a+') as f:
        #     f.write('loss:{:.4f}\n'.format(loss.cpu().data.numpy()))

    def train_proto(self,module):
        dataiter = self.df.train_feeder()
        C = tr.FloatTensor(self.df.prototype_feeder(self.nn_config['k_shot']))
        optim = self.optimizer(module)
        for X,y_ in dataiter:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X,y_,C = X.cuda(),y_.cuda(),C.cuda()
            optim.zero_grad()
            score = module.forward_softmax(tr.autograd.Variable(X,requires_grad=False),
                                           tr.autograd.Variable(C,requires_grad=False))
            loss = module.cross_entropy_loss(score,tr.autograd.Variable(y_,requires_grad=False))
            loss.backward()
            optim.step()


    def prediction(self,score):
        pred_labels = np.argmax(score,axis=1).astype('float32')
        return pred_labels

    def metrics(self,true_labels,pred_labels):
        true_labels =list(true_labels)
        pred_labels = list(pred_labels)
        f1 = sklearn.metrics.f1_score(y_true=true_labels,y_pred=pred_labels, average='macro')
        accuracy = sklearn.metrics.accuracy_score(y_true=true_labels,y_pred=pred_labels)
        return f1,accuracy

    def test(self,module):
        f=open(self.nn_config['report_filePath'],'a+')
        test_data =self.df.test_feeder()
        C = tr.FloatTensor(self.df.prototype_feeder(self.nn_config['k_shot']))

        for X, y_ in test_data:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X,y_,C = X.cuda(),y_.cuda(),C.cuda()

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