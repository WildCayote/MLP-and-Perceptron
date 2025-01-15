from typing import Callable
import numpy as np


class MultiLayerPerceptron:
    def __init__(self, num_inputs:int, num_hidden:int, hidden_width:int, activation_function:Callable[[np.ndarray], np.ndarray], activation_derivated:Callable[[np.ndarray], np.ndarray],  num_output:int = 1, learning_rate:float = 0.1):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.hidden_width = hidden_width
        self.num_output = num_output
        self.learning_rate = np.array(learning_rate)
        self.activation_function = activation_function
        self.activation_derviated = activation_derivated

        # initialize the weights
        self.weights = self.initialize_weights()

        # initialize the biases
        self.biases = self.initialize_biases()

    def initialize_weights(self):
        weights = []
        # define the input weights
        if self.num_hidden > 0:
            weights.append(np.random.rand(self.num_inputs, self.hidden_width))
        else:
            weights.append(np.random.rand(self.num_inputs, self.num_output))

        # define the weights in the hidden layers
        if self.num_hidden > 1:
            for _ in range(self.num_hidden - 1):
                weights.append(
                    np.random.rand(self.hidden_width, self.hidden_width)
                )
        
        # define the weights between the last hidden layer and the output layer
        if self.num_hidden > 0:
            weights.append(np.random.rand(self.hidden_width, self.num_output))

        return weights 

    def initialize_biases(self):
        biases = []
        # define the biases in the input layer
        if self.num_hidden > 0:
            biases.append(np.random.rand(self.hidden_width))
        else: 
            biases.append(np.random.rand(self.num_output))

        # defin the biases between the hidden layers
        if self.num_hidden > 1:
            for _ in range(self.num_hidden - 1):
                biases.append(
                    np.random.rand((self.hidden_width))
                )

        # define the biases between the last hidden layer and the output layer
        if self.num_hidden > 0:
            biases.append(np.random.rand(self.num_output))

        return biases

    def predict(self, input:np.ndarray):
        # list for holding activation values for each layer
        activations = [input]

        for weight in self.weights:
            result = self.activation_function(np.dot(weight.T, input))
            activations.append(result)
            input = result

        return result, activations

    def loss(self, predictions:np.ndarray, true_values:np.ndarray):
        error = -np.sum(true_values * np.log(predictions + 1e-12) / true_values.shape[0])
        return error

    def train(self, train_x:np.ndarray, train_y:np.ndarray):
        # forward propagation
        prediction, activations=self.predict(input=train_x)
        error = self.loss(predictions=prediction, true_values=train_y)

        # backpropagation
        weight_slopes = [None] * len(self.weights)
        for i in reversed(range(len(self.weights))):
            activation = activations[i + 1]
            delta_intermidiate = error * self.activation_derviated(activation)
            current_activation = activations[i]
            weight_slopes[i] = np.dot(current_activation, delta_intermidiate.T)

        return weight_slopes
            
    def fit(self,  train_x:np.ndarray, train_y:np.ndarray, num_epochs:int = 20):
        pass

if __name__ == '__main__':
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    test_x = np.array([[-1,2,3], [-1,-2,3], [-1,2,-3]]) 
    test_y = np.array([1, 0, 0])
    mlp = MultiLayerPerceptron(num_inputs=3, num_hidden=2, num_output=1, hidden_width=4, activation_function=sigmoid, activation_derivated=sigmoid_derivative)

    mlp.train(train_x=test_x, train_y=test_y)
