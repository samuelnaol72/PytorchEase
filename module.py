#importing necessary modules
import numpy as np
from .tensor import Tensor

class Module:
    def __init__(self):
        self.training = True
        self._parameters = dict()
        self._modules = dict()

    def __setattr__(self, name, value):
        super(Module, self).__setattr__(name, value)
        if isinstance(value, Tensor) and value.requires_grad:
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

    def modules(self):
        yield self
        for name, module in self._modules.items():
            if module is None:
                continue
            for m in module.modules():
                yield m
        
    def parameters(self):
        modules = self.modules()
        for module in modules:
            members = module._parameters.items()
            for k, v in members:
                yield v

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = None

    def update(self, lr):
        for parameter in self.parameters():
            parameter -= lr * parameter.grad

    def train(self):
        for module in self.modules():
            module.training = True

    def eval(self):
        for module in self.modules():
            module.training = False

    def forward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)