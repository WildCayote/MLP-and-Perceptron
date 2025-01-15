from typing import Callable, List
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

    def calculate_gradients(self, train_x:np.ndarray, train_y:np.ndarray):
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
        
        bias_slopes = [None] * len(self.biases)
        for i in reversed(range(len(self.biases))):
            activation = activations[i + 1]
            delta_intermidiate = error * self.activation_derviated(activation)
            delta_intermidiate = np.dot(activation, delta_intermidiate.T)
            bias_reshaped = self.biases[i].reshape(-1, self.biases[i].shape[0])
            bias_slopes[i] = np.dot(bias_reshaped, delta_intermidiate)

        return weight_slopes, bias_slopes,  error

    def update_weights(self, weight_slopes: List[np.ndarray]):
        new_weights = []
        for weight, gradient in zip(self.weights, weight_slopes):
            updated_weight = weight + self.learning_rate * gradient
            new_weights.append(updated_weight)
        
        self.weights = new_weights
    
    def update_biases(self, bias_slopes: List[np.ndarray]):
        # print(self.biases, ' before update')
        new_biases = []
        for bias, gradient in zip(self.biases, bias_slopes):
            updated_bias = bias + self.learning_rate * gradient
            new_biases.append(updated_bias.reshape(-1, updated_bias.shape[0]).sum(axis=1))
        
        self.biases = new_biases
        # print(self.biases, ' after update')

    def fit(self,  train_x:np.ndarray, train_y:np.ndarray, num_epochs:int = 1, verbose:bool = False):
        for epoch in range(num_epochs):
            # make predictions, calculate errors and calculate gradients
            weight_gradients, bias_gradients, error = self.calculate_gradients(train_x=train_x, train_y=train_y)

            # update the weights
            self.update_weights(weight_slopes=weight_gradients)
            
            # update the biases
            self.update_biases(bias_slopes=bias_gradients)

            if verbose: print(f'Loss at epoch {epoch + 1}: {error}')

if __name__ == '__main__':
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        s = sigmoid(x)
        return s * (1 - s)

    test_x = np.array([[-1,2,3], [-1,-2,3], [-1,2,-3]]) 
    test_y = np.array([1, 0, 0])
    mlp = MultiLayerPerceptron(num_inputs=3, num_hidden=2, num_output=1, hidden_width=4, activation_function=sigmoid, activation_derivated=sigmoid_derivative)

    mlp.fit(train_x=test_x, train_y=test_y, verbose=True, num_epochs=100)
