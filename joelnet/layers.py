"""
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from typing import Dict, Callable, Tuple, Sequence

import numpy as np

from joelnet.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError

class Linear(Layer):
    """
    computes output = inputs @ w + b
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        # inputs will be (batch_size, input_size)
        # outputs will be (batch_size, output_size)
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T

F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    An activation layer just applies a function
    elementwise to its inputs
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def relu(x: Tensor) -> Tensor:
    return x * (x > 0)


def relu_prime(x: Tensor) -> Tensor:
    return 1 * (x > 0)


class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)


def construct_layers(linear_layer_sizes: Tuple[int], activation: Activation) -> Sequence[Layer]:
    layers = []

    for i, current_layer_size in enumerate(linear_layer_sizes):
        next_layer_size = linear_layer_sizes[i + 1]
        layers.append(
            Linear(input_size=current_layer_size, output_size=next_layer_size))

        # we don't want to append activation function on the last layer
        # so we break out of the loop when we are on the penultimate layer
        if i == len(linear_layer_sizes) - 2:
            break
        
        layers.append(activation())
    
    return layers
