"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from joelnet.nn import NeuralNet
from joelnet.layers import Tanh, Relu
from joelnet.optim import Adam, SGD

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet(
    hidden_layer_sizes=(10, ),
    input_size=2,
    output_size=2,
    activation=Relu
)

net.fit(inputs, targets, optimizer=SGD())

for x, y in zip(inputs, targets):
    predicted = net.predict(x)
    print(x, predicted, y)
