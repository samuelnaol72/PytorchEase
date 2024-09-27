import numpy as np
class Tensor(np.ndarray):

    def __new__(cls, input_array, requires_grad=False):
        obj = np.asarray(input_array).view(cls)
        obj.grad = None
        obj.requires_grad = requires_grad
        lambda_fn = lambda x: 0 if x is None else x #The lambda function takes a single argument x, and if x is None, it returns 0, otherwise, it returns x.
        obj.backward_fn = lambda leaf_grad: setattr(obj, 'grad', lambda_fn(obj.grad) + leaf_grad) if requires_grad else lambda *args: None
        # 
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.grad = getattr(obj, 'grad', None)
        self.requires_grad = getattr(obj, 'requires_grad', None)
        self.backward_fn = getattr(obj, 'backward_fn', None)

    def backward(self, *args):
        self.backward_fn(*args)