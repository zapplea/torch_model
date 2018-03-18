"""
This script provides a tutorial for basic PyTorch operations
Please refer to Appendix in Lab2.pdf for more details.
"""

# import libraries
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

"""
1. Tensors
"""

"""
1.1 What are Tensors?
Tensors are similar to NumPy's ndarrays, with the addition that
Tensors can also be used on a GPU to accelerate computing.
"""
print('-'*50)
print('1.1 Construct a tensor')
# construct a randomly initialised tensor matrix
x = torch.rand(5, 3)
print('a randomly initialised tensor matrix is ', x)
# get matrix size
print('the size of x is', x.size())

"""
1.2 Tensors operations
Tensors support basic operations such as addition, subtraction,
multiplication, and division.
There are multiply ways of specifying operations. In the following
example, we will take a look at the addition operation.
"""
print('-'*50)
print('1.2 Tensors operations')

# Addition: method 1
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print('*'*10, 'Addition: syntax 1', '*'*10)
print('the sum of x and y is', x + y)

# Addition: method 2
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print('*'*10, 'Addition: syntax 2', '*'*10)
print('the sum of x and y is', torch.add(x, y))

# Addition: in-place
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print('*'*10, 'Addition: in-place', '*'*10)
print('before addition, y is ', y)
y.add_(x)
print('after in-place addition, y is ', y)

"""
Indexing is also NumPy-like.
"""
print('*'*10, 'indexing', '*'*10)
# access the second column
print('second column of x is', x[:, 1])
# access the first row
print('first row of x is ', x[0, :])


"""
1.3 Tensors <=> NumPy's ndarrays
Tensors can be converted to NumPy's ndarrays, and can be formed
by NumPy's ndarrays.
NOTE! The Torch Tensor and NumPy array will share their underlying
memory locations, and changing one will change the other.
"""
print('-'*50)
print('1.3 Tensors <-> NumPy\'s ndarrays')

# Tensors -> Numpy arrays
# create a randomly initialized tensor matrix
print('*'*10, 'Tensor -> NumPy', '*'*10)
x = torch.rand(5, 3)
print('a tensor is ', x)
# convert tensors to numpy array
y = x.numpy()
print('the equivalent numpy array is \n', y, '\n')

# Tensors <- Numpy arrays
print('*'*10, 'Tensor <- NumPy', '*'*10)
# create a numpy array
x = np.array([[3, 4], [3, 5]])
print('a numpy array is \n', x, '\n')
# convert numpy array to tensor
y = torch.from_numpy(x)
print('the equivalent tensor is ', y)

# Tensors and Numpy arrays share memory locations
print('*'*10, 'Tensors and Numpy arrays share memory', '*'*10)
a = torch.ones(5)
b = a.numpy()
a.add_(1)
print(a)
print(b)

"""
2. Autograd: automatic differentiation
Central to all neural networks in PyTorch is the autograd package.
It provides automatic differentiation for all operations on Tensors.
"""

"""
2.1 Variable
A Variable wraps a Tensor, and can have gradients computed automatically
by calling .backward()
"""
print('-'*50)
print('2.1 Variables')
# Create variables
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)

# Define a function
y = w * x + b        # y = 2 * x + 3

# Compute gradients.
y.backward()        # equal to y.backward(torch.Tensor([1.0]))

# Print out the gradients.
print('dy/dx: {}'.format(x.grad.data))    # x.grad = 2
print('dy/dw: {}'.format(w.grad.data))    # w.grad = 1
print('dy/db: {}'.format(b.grad.data))    # b.grad = 1

"""
2.2 Gradient
The following section shows an example of how to get grad_fn and how 
to get gradients.
"""
print('-'*50)
print('2.2 Gradient')
x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
x = Variable(x, requires_grad=True)
y = x+2
z = y * y * 3
out = z.mean()

print('x:', x, 'y:', y, 'z:', z, 'out:', out)

print('grad_fn of x:', x.grad_fn)
print('grad_fn of y:', y.grad_fn)
print('grad_fn of z:', z.grad_fn)
print('grad_fn of out:', out.grad_fn)

# # please comment 'z.backward(...)' and 'y.backward(..)',
# # and uncomment the following two lines to get dout/dx
out.backward(torch.Tensor([1]))
print('dout/dx:', x.grad.data)

# please comment 'out.backward(...)' and 'y.backward(..)',
# and uncomment the following two lines to get dz/dx
# z.backward(torch.Tensor([[1, 1, 1], [1, 1, 1]]))
# print('dz/dx:', x.grad.data)

# please comment 'out.backward(...)' and 'z.backward(..)',
# and uncomment the following two lines to get dy/dx
# y.backward(torch.Tensor([[1, 1, 1], [1, 1, 1]]))
# print('dy/dx:', x.grad.data)

"""
3. Neural Network
The following section demonstrates how to build and train a
simple neural network with 4 inputs, 1 hidden layer (with 2
hidden neurons), and 1 output neuron.
"""
print('-'*50)
print('3 Neural Network')

input_neurons = 4
hidden_neurons = 2
output_neurons = 1


# define a simple neural network with one sigmoid hidden layer
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
loss_func = torch.nn.MSELoss()

# create some random inputs
input = Variable(torch.randn(10, 4))
# give a random target
target = Variable(torch.randn(10, 1))

# start training
for epoch in range(10):
    # perform forward pass to calculate the actual output
    output = net(input)

    # calculate the errors
    loss = loss_func(output, target)
    print('Epoch [%d/10] Loss: %.4f' % (epoch + 1, loss.data[0]))

    # clear gradient buffers of all parameters
    net.zero_grad()

    # perform backward pass: compute gradients of the loss with respect to
    # all the learnable parameters of the model .
    loss.backward()

    # define learning rate
    learning_rate = 0.01
    # define optimiser
    optimiser = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # update optimiser
    optimiser.step()


