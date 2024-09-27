#importing the necessary class 
from .module import Module
import numpy as np
import math
from .tensor import Tensor

class Linear(Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_parameters()

    def init_parameters(self):
        W = np.random.randn(self.in_features, self.out_features) * math.sqrt(2 / (6 * self.in_features))
        bound = 1 / math.sqrt(self.in_features)
        b = np.random.uniform(-bound, bound, size=(self.out_features))
        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)

    def forward(self, x):
        return wx_plus_b(self.W, self.b, x)


def wx_plus_b(W, b, input):
    output = np.zeros((input.shape[0], W.shape[1]))
    for i in range(len(input)):
      output[i,:]= np.matmul(input[i,:], W) + b
    output =Tensor(output)
    def backward(grad_output):
        b_grad = np.sum(grad_output, axis=0)
        W_grad = np.matmul(input.T,grad_output)/input.shape[0]
        input_grad = np.matmul(grad_output, W.T)
  
        # to store the new gradient for the update later
        W.backward_fn(W_grad)
        b.backward_fn(b_grad/input.shape[0])
        #backpropagating the new calculate gradient
        input.backward_fn(input_grad)

    output.backward_fn = backward
    return output