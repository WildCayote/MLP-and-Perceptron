import numpy as np

class Perceptron:
    def __init__(self, num_inputs:int, learning_rate:float = 0.1):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate

        # initialize the weights
        self.weights = self.initialize_weights()
        self.bias = self.initialize_bias()

    def initialize_weights(self):
        return np.random.rand(self.num_inputs)

    def initialize_bias(self):
        return np.random.rand(1)

    def predict(self):
        pass
    
    def loss(self):
        pass

    def train(self):
        pass
