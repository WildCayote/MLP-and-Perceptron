from typing import Callable
import numpy as np


class MultiLayerPerceptron:
    def __init__(self, num_inputs:int, num_hidden:int, hidden_width:int, activation_function:Callable[[np.ndarray], np.ndarray], num_output:int = 1, learning_rate:float = 0.1):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.hidden_width = hidden_width
        self.num_output = num_output
        self.learning_rate = np.array(learning_rate)
        self.activation_function = activation_function

        # initialize the weights
        self.input_weights, self.hidden_weights, self.output_weights = self.initialize_weights()

        # initialize the biases
        self.input_biases, self.hidden_biases, self.output_biases = self.initialize_biases()

    def initialize_weights(self):
        # define the input weights
        input_weights = np.random.rand(self.num_inputs, self.hidden_width)

        # define the weights in the hidden layers
        hidden_weights = []
        for _ in range(self.num_hidden - 1):
            hidden_weights.append(
                np.random.rand(self.hidden_width, self.hidden_width)
            )
        hidden_weights = np.array(hidden_weights)

        # define the weights between the last hidden layer and the output layer
        output_weights = np.random.rand(self.hidden_width, self.num_output)

        return input_weights, hidden_weights, output_weights

    def initialize_biases(self):
        # define the biases in the input layer
        input_biases = np.random.rand(self.hidden_width)

        # defin the biases between the hidden layers
        hidden_biases = []
        for _ in range(self.num_hidden - 1):
            hidden_biases.append(
                np.random.rand((self.hidden_width))
            )
        hidden_biases = np.array(hidden_biases)

        # define the biases between the last hidden layer and the output layer
        output_biases = np.random.rand(self.num_output)

        return input_biases, hidden_biases, output_biases

    def predict(self, input:np.ndarray):
        # input layer
        z1 = input @ self.input_weights + self.input_biases
        z1 = self.activation_function(z1)

        # hidden layers
        z2 = None
        for layer_idx in range(self.hidden_weights.shape[0]):
            weights = self.hidden_weights[layer_idx]
            biases = self.hidden_biases[layer_idx]

            if type(z2) != np.ndarray:
                result = z1 @ weights + biases
            else:
                result = z2 @ weights + biases
            z2 = self.activation_function(result)
        
        # output layer
        z3 = z2 @ self.output_weights + self.output_biases
        z3 = self.activation_function(z3)

        return z3

    def train(self):
        pass

if __name__ == '__main__':
    def heaviside_step_func(input: np.ndarray):
        return (input > 0).astype(int)

    test_x = np.array([[-1,2,3], [-1,-2,3], [-1,2,-3], [0,2,3]]) 
    test_y = np.array([1, 0, 0, 1])
    mlp = MultiLayerPerceptron(num_inputs=3, num_hidden=2, num_output=1, hidden_width=4, activation_function=heaviside_step_func)
