#importing necessary class
from .module import Module
import numpy as np

class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return relu_function(x)


def relu_function(input):
    output = np.maximum(input, 0)
    def backward(grad_output):
        localgrad = np.where(input > 0, 1, 0)
        grad = grad_output * localgrad
        input.backward_fn(grad)
    output.backward_fn = backward
    return output