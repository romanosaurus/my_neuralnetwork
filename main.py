import numpy as np

from NeuralNetwork import NeuralNetwork
from Utils import sigmoid, sigmoid_derivative

if __name__ == "__main__":
    X = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y, sigmoid, sigmoid_derivative)

    for i in range(20000):
        nn.feedforward()
        nn.backprop()

print(nn.output)
