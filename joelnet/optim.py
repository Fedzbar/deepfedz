"""
We use an optimizer to adjust the parameters
of our network based on the gradients computed
during backpropagation
"""
from joelnet.tensor import Tensor
import numpy as np

class Optimizer:
    def step(self, params: Tensor, grads: Tensor) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, params: Tensor, grads: Tensor) -> None:
        param -= self.lr * grad

class Adam(Optimizer): 
    def __init__(self, lr: float = 0.001, beta_1: float = 0.9, beta_2: float = 0.999, epsilon: float = 10**-8):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = 0.
        self.v = 0.

        self.t = 0

    def set_dimensions(self, params: Tensor) -> None: 
        self.m = [np.zeros_like(param) for param in params]
        self.v = [np.zeros_like(param) for param in params]

    def step(self, params: Tensor, grads: Tensor) -> None: 
        self.t += 1

        # update first raw moment estimate
        self.m = [self.beta_1 * m + (1 - self.beta_1) * grad for m, grad in zip(self.m, grads)]
        # update second raw moment estimate
        self.v = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2) for v, grad in zip(self.v, grads)]

        # bias corrected first raw moment
        m_hats = [m / (1 - (self.beta_1 ** self.t)) for m in self.m]
        # bias corrected second raw moment
        v_hats = [v / (1 - (self.beta_2 ** self.t)) for v in self.v]

        for i, param in enumerate(params):
            param -= self.lr * m_hats[i] / (np.sqrt(v_hats[i]) + self.epsilon)