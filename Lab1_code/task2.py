"""
Sometimes you will want to specify models that are more complex than a
sequence of existing Modules.
This script provides an example of building a neural network with custom
nn modules.
"""

"""
Step 0: Download and import all required libraries

To run this script, you need to have the following packages installed:
- torch: a python deep learning package
- pandas: a python data analysis package; if you are familiar with numpy, 
you can use numpy instead

To install pytorch, please follow the instructions on http://pytorch.org/

To install pandas, in your terminal, type `pip3 install pandas` for python 3
"""

# import libraries
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn.functional as F

"""
Step 1: Load and setup training dataset

The dataset is separated into two files from original dataset:
    iris_train.csv = dataset for training purpose, 80% from the original data
    iris_test.csv  = dataset for testing purpose, 20% from the original data
"""

# load training data
data_train = pd.read_csv('dataset/iris_train.csv')

# convert string target values to numeric values
#       class 0: Iris-setosa
#       class 1: Iris-versicolor
#       class 2: Iris-virginica
data_train.at[data_train['species'] == 'Iris-setosa', ['species']] = 0
data_train.at[data_train['species'] == 'Iris-versicolor', ['species']] = 1
data_train.at[data_train['species'] == 'Iris-virginica', ['species']] = 2
data_train = data_train.apply(pd.to_numeric)

# convert pandas dataframe to array
# the first 4 columns are features
# the last column is target
data_train_array = data_train.as_matrix()

# split x (features) and y (targets)
x_array = data_train_array[:, :4]
y_array = data_train_array[:, 4]

# create Tensors to hold inputs and outputs, and wrap them in Variables,
# as Torch only trains neural network on Variables
#
X = Variable(torch.Tensor(x_array).float())
Y = Variable(torch.Tensor(y_array).long())

"""
Step 2: Define and train a neural network

Here we build a neural network with one hidden layer.
    input layer: 4 neurons, representing the features of Iris
    hidden layer: 10 neurons, using Sigmoid as activation function
    output layer: 3 neurons, representing the classes of Iris
    
The network will be trained with Stochastic Gradient Descent (SGD) as an 
optimiser, that will hold the current state and will update the parameters
based on the computed gradients.

Its performance will be evaluated using cross-entropy.
    
We implement the same neural network using a custom Module subclass
"""

# define the number of neurons for input layer, hidden layer and output layer
# define learning rate and number of epoch on training
input_neurons = x_array.shape[1]
hidden_neurons = 10
output_neurons = np.unique(y_array).size
learning_rate = 0.01
num_epoch = 500


# define a customised neural network structure
class TwoLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        # get hidden layer input
        h_input = self.hidden(x)
        # define activation function for hidden layer
        h_output = F.sigmoid(h_input)
        # get output layer output
        y_pred = self.out(h_output)

        return y_pred

# define a neural network using the customised structure
net = TwoLayerNet(input_neurons, hidden_neurons, output_neurons)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# train a neural network
for epoch in range(num_epoch):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)

    # Compute loss
    loss = loss_func(Y_pred, Y)
    all_losses.append(loss.data[0])

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(F.softmax(Y_pred), 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epoch, loss.data[0], 100 * sum(correct)/total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss

# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(all_losses)
# plt.show()

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every iris flower (rows)
which class the network guesses (columns). 

"""

confusion = torch.zeros(output_neurons, output_neurons)

Y_pred = net(X)
_, predicted = torch.max(F.softmax(Y_pred), 1)

for i in range(x_array.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

"""
Step 3: Load and setup testing dataset

The dataset is separated into two files from original dataset:
    iris_train.csv = dataset for training purpose, 80% from the original data
    iris_test.csv  = dataset for testing purpose, 20% from the original data
"""

# load testing data
data_test = pd.read_csv('dataset/iris_test.csv')

# convert string target values to numeric values
#       class 0: Iris-setosa
#       class 1: Iris-versicolor
#       class 2: Iris-virginica
data_test.at[data_test['species'] == 'Iris-setosa', ['species']] = 0
data_test.at[data_test['species'] == 'Iris-versicolor', ['species']] = 1
data_test.at[data_test['species'] == 'Iris-virginica', ['species']] = 2
data_test = data_test.apply(pd.to_numeric)

# convert pandas dataframe to array
# the first 4 columns are features
# the last column is target
data_test_array = data_test.as_matrix()

# split x (features) and y (targets)
x_test_array = data_test_array[:, :4]
y_test_array = data_test_array[:, 4]

# create Tensors to hold inputs and outputs, and wrap them in Variables,
# as Torch only trains neural network on Variables
X_test = Variable(torch.Tensor(x_test_array).float())
Y_test = Variable(torch.Tensor(y_test_array).long())

"""
Step 4: Test the neural network

Pass testing data to the built neural network and get its performance
"""
# test the neural network using testing data
# It is actually performing a forward pass computation of predicted y
# by passing x to the model.
# Here, Y_pred_test contains three columns, where the index of the
# max column indicates the class of the instance
Y_pred_test = net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(F.softmax(Y_pred_test), 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every iris flower (rows)
which class the network guesses (columns). 

"""

confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(x_test_array.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)
