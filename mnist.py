# This first part is from:

# Author: Arthur Mensch <arthur.mensch@m4x.org>
# License: BSD 3 clause

import pickle
from sklearn.metrics import accuracy_score
from joelnet.optim import SGD, Adam
from joelnet.layers import Tanh, Relu
from joelnet.nn import NeuralNet
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np

train_samples = 10000

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# labels come as ['1', '5', ...], we want to convert them to a one hot matrix structure

def string_array_to_one_hot_matrix(to_convert: np.ndarray) -> np.ndarray:
    # matrix size: number of labels * 10 options (numbers 0 - 9)
    matrix = np.zeros((to_convert.size, 10))
    matrix[np.arange(to_convert.size), to_convert.astype(int)] = 1
    return matrix


y_train = string_array_to_one_hot_matrix(y_train)
y_test = string_array_to_one_hot_matrix(y_test)

net = NeuralNet(
    hidden_layer_sizes=(300, 100 ),
    activation=Tanh,
    input_size=784,
    output_size=10
)

net.fit(
    X_train,
    y_train,
    optimizer=Adam(),
    num_epochs=1000)


def get_max_of_matrix_per_row(matrix: np.ndarray) -> np.ndarray:
    max_matrix = np.zeros_like(matrix)
    max_matrix[np.arange(len(matrix)), matrix.argmax(1)] = 1
    return max_matrix


y_pred = net.predict(X_test)
y_pred = get_max_of_matrix_per_row(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
pickle.dump(net, open('models/mnist_net.p', 'wb'))