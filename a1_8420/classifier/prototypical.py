import torch as tr
import torch.nn.functional as F
import numpy as np
import sklearn.metrics

class Net(tr.nn.Module):
    def __init__(self,nn_config):
        super(Net,self).__init__()
        self.nn_config = nn_config
        in_dim = self.nn_config['feature_dim']
        out_dim = self.nn_config['layer_dim'][0]
        self.linear1 = tr.nn.Linear(in_dim,out_dim)

    def forward_nonlinear(self,X):
        linear_layer1 = self.linear1(X)
        hidden_layer = F.relu(linear_layer1)
        return hidden_layer

    def forward_sample(self,X):
        return self.forward_nonlinear(X)

    def forward_prot(self,C):
        return self.forward_nonlinear(C)

    def forward_softmax(self,X,C):

        X = self.forward_sample(X)
        # shape = (batch size, 1,feature dim)
        X = tr.unsqueeze(X,dim=1)
        # shape = (batch size, labels num,feature dim)
        X = X.repeat(1,self.nn_config['label_dim'],1)
        C = self.forward_prot(C)
        # shape = (batch size, labels num)
        euclidean_distance = tr.sqrt(tr.mul(tr.add(X,-C),tr.add(X,-C)).sum(2))
        score = F.softmax(euclidean_distance)
        return score

    def cross_entropy_loss(self,input,target):
        input = tr.log(input)
        loss = tr.nn.NLLLoss(size_average=True,reduce=True)
        return loss(input,target)

class PrototypicalNet:
    def __init__(self,nn_config,df):
        self.nn_config = nn_config
        self.df = df

    def optimizer(self,model):
        optim = tr.optim.SGD(model.parameters(),lr=self.nn_config['lr'],weight_decay=self.nn_config['weight_decay'])
        return optim

    def classifier(self):
        with tr.cuda.device(self.nn_config['gpu']):
            module = Net(self.nn_config)
            if self.nn_config['cuda'] and tr.cuda.is_available():
                module.cuda()
            for i in range(self.nn_config['epoch']):
                self.train(module)
                self.test(module)

    def train(self,module):
        dataiter = self.df.train_feeder()
        C = tr.FloatTensor(self.df.prototype_feeder(self.nn_config['k_shot']))
        optim = self.optimizer(module)

        for X,y_ in dataiter:
            if self.nn_config['cuda'] and tr.cuda.is_available():
                X,y_,C = X.cuda(),y_.cuda(),C.cuda()
            optim.zero_grad()
            score = module.forward_softmax(tr.autograd.Variable(X,requires_grad=False),tr.autograd.Variable(C,requires_grad=False))
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

            pred_labels = self.prediction(score.numpy())
            f1,accuracy = self.metrics(true_labels=y_.numpy().astype('float32'),pred_labels=pred_labels)
            f.write('Test: loss:{:.4f}, accuracy:{:.4f}, f1:{:.4f}\n'.format(loss.data.numpy(), f1, accuracy))
            f.flush()
        f.close()
