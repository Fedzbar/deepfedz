"""
A NeuralNet is just a collection of layers.
It behaves a lot like a layer itself, although
we're not going to make it one.
"""
from typing import Sequence, Iterator, Tuple

from joelnet.tensor import Tensor
from joelnet.layers import Layer, Activation, Relu, construct_layers
from joelnet.loss import Loss, MSE
from joelnet.optim import Optimizer, SGD, Adam
from joelnet.data import DataIterator, BatchIterator


class NeuralNet:
    def __init__(self, hidden_layer_sizes: Tuple[int], input_size: int,
                 output_size: int, activation: Activation = Relu()) -> None:
        linear_layer_sizes = (input_size,) + \
            hidden_layer_sizes + (output_size,)
        self.layers: Sequence[Layer] = construct_layers(
            linear_layer_sizes, activation)

    def fit(self, inputs: Tensor, targets: Tensor, num_epochs: int = 5000,
            iterator: DataIterator = BatchIterator(), loss: Loss = MSE(),
            optimizer: Optimizer = Adam()) -> None:

        for epoch in range(num_epochs):
            epoch_loss = 0.
            for batch in iterator(inputs, targets):
                predicted = self.forward(batch.inputs)
                epoch_loss += loss.loss(predicted, batch.targets)
                grad = loss.grad(predicted, batch.targets)
                self.backward(grad)
                self.step(optimizer)

                print(f"{epoch}: {epoch_loss}")

    def predict(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def step(self, optimizer) -> None:
        if isinstance(optimizer, SGD):
            for param, grad in self.params_and_grads():
                optimizer.step(param, grad)

        if isinstance(optimizer, Adam):
            params = [param for param, grad in self.params_and_grads()]
            grads = [grad for param, grad in self.params_and_grads()]

            optimizer.set_dimensions(params)

            optimizer.step(params, grads)
