import numpy as np
from .tensor import Tensor
class CrossEntropyLoss:
    def __call__(self, input, target):
        return compute_cross_entropy_loss(input, target)
def compute_cross_entropy_loss(input, target):
    # input here is the one fed to softmax
    log_softmax = log_softmax_func(input)
    loss = negative_log_likelihood_loss(log_softmax, target)
    
    def backward():
        soft_output = softmax(input)
        grad = soft_output.copy()
        batch_size = grad.shape[0]
        for i, t in enumerate(target):
            grad[i, t] -= 1
        input.backward_fn(Tensor(grad))
    output = Tensor(loss, requires_grad=True)
    output.backward_fn = backward
    return output
def log_softmax_func(input):
    log_sum_exp = np.log(np.sum(np.exp(input), axis=1, keepdims=True))
    log_softmax = input - log_sum_exp
    return log_softmax
def negative_log_likelihood_loss(log_softmax, target):
    batch_size = log_softmax.shape[0]
    negative_log_likelihood = -sum([log_softmax[i, target[i]] for i in range(batch_size)]) / batch_size
    return negative_log_likelihood

def softmax(input):
  exp = np.exp(input)
  denominator = np.sum(exp, axis=1, keepdims=True)
  return exp/denominator