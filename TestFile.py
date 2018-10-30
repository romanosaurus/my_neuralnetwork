import numpy as np

from NeuralNetwork import NeuralNetwork

if __name__ == "__main__":
    X = np.array([[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    nn = NeuralNetwork(X, y)

    for i in range(20000):
        nn.feedforward()
        nn.backprop()

print(nn.output)
