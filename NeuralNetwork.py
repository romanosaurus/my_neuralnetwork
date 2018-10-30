import numpy as np

from Utils import sigmoid, sigmoid_derivative


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        '''
             on essaye de déterminer l'output
             -> np.dot() permet l'addition de deux tableaux
                -> on calcule alors à partir des poids la couche intermédiaire pour l'output
                -> on determine ensuite l'output de notre feedforward
        '''
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        '''
            Ajustement des poids
                -> Utilisation d'une fonction pour déterminer la perte
                   elle détermine l'ajout à faire sur les poids, pour ajuster notre apprentissage
                -> c'est une somme de la différence entre les valeurs déterminées durant le feed forward
                et la valeur attendue y
        '''

        d_weights2 = np.dot(
            self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(
            self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        self.weights1 += d_weights1
        self.weights2 += d_weights2
