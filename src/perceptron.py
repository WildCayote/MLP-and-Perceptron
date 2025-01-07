from typing import Callable
import numpy as np

class Perceptron:
    def __init__(self, num_inputs:int, activation_function: Callable[[np.ndarray], np.ndarray], learning_rate:float = 0.1):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.activation_function = activation_function

        # initialize the weights
        self.weights = self.initialize_weights()
        self.bias = self.initialize_bias()

    def initialize_weights(self):
        return np.random.rand(self.num_inputs)

    def initialize_bias(self):
        return np.random.rand(1)

    def predict(self, input:np.ndarray):
        # multiply the inputs with the weights
        z = input @ self.weights.T + self.bias
        
        # pass the result through an activation function
        predictions = []
        try:
            for batch in input:
                predictions.append(
                    self.activation_function(batch)
                )
        except Exception as e:
            return self.activation_function(input)
        return predictions
    
    def loss(self, predictions: np.ndarray, true_values: np.ndarray):
        error = predictions - true_values
        return error

    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        prediction = self.predict(input=train_x)
        error = self.loss(predictions=prediction, true_values=train_y)

        # update the weights
        self.weights += self.learning_rate * error * train_x

        # update the bias
        self.bias += self.weights * error


if __name__ == '__main__':
    def heaviside_step_func(input: np.ndarray):
        if input > 0: return np.array(1)
        return np.array(0)

    test_x = np.array([[-1,2,3], [-1,-2,3], [-1,2,-3], [0,2,3]]) 
    test_y = [1, 0, 0, 1]
    perceptron = Perceptron(num_inputs=3, activation_function=heaviside_step_func)