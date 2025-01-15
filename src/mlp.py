from typing import Callable, List
import numpy as np


class MultiLayerPerceptron:
    def __init__(self, num_inputs, num_hidden, hidden_width, activation_function, activation_derivated, num_output=1, learning_rate=0.1):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.hidden_width = hidden_width
        self.num_output = num_output
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.activation_derviated = activation_derivated

        # Initialize weights and biases
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    def initialize_weights(self):
        weights = []
        if self.num_hidden > 0:
            weights.append(np.random.rand(self.num_inputs, self.hidden_width))
            for _ in range(self.num_hidden - 1):
                weights.append(np.random.rand(self.hidden_width, self.hidden_width))
            weights.append(np.random.rand(self.hidden_width, self.num_output))
        else:
            weights.append(np.random.rand(self.num_inputs, self.num_output))
        return weights

    def initialize_biases(self):
        biases = []
        if self.num_hidden > 0:
            biases.append(np.random.rand(self.hidden_width))
            for _ in range(self.num_hidden - 1):
                biases.append(np.random.rand(self.hidden_width))
            biases.append(np.random.rand(self.num_output))
        else:
            biases.append(np.random.rand(self.num_output))
        return biases

    def predict(self, input):
        activations = [input]
        for weight, bias in zip(self.weights, self.biases):
            input = self.activation_function(np.dot(input, weight) + bias)
            activations.append(input)
        return input, activations

    def loss(self, predictions, true_values):
        return -np.mean(true_values * np.log(predictions + 1e-12))

    def calculate_gradients(self, train_x, train_y):
        predictions, activations = self.predict(train_x)
        error = self.loss(predictions, train_y)
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)

        delta = (predictions - train_y) * self.activation_derviated(activations[-1])
        for i in reversed(range(len(self.weights))):
            weight_gradients[i] = np.dot(activations[i].T, delta)
            bias_gradients[i] = np.sum(delta, axis=0)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derviated(activations[i])

        return weight_gradients, bias_gradients, error

    def update_weights(self, weight_slopes):
        self.weights = [w - self.learning_rate * g for w, g in zip(self.weights, weight_slopes)]

    def update_biases(self, bias_slopes):
        self.biases = [b - self.learning_rate * g for b, g in zip(self.biases, bias_slopes)]

    def fit(self, train_x, train_y, num_epochs=1, verbose=False):
        for epoch in range(num_epochs):
            weight_gradients, bias_gradients, error = self.calculate_gradients(train_x, train_y)
            self.update_weights(weight_gradients)
            self.update_biases(bias_gradients)
            if verbose:
                print(f'Epoch {epoch + 1}, Loss: {error}')

if __name__ == '__main__':
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(x):
        return x * (1 - x)

    test_x = np.array([[0.5, 0.2, 0.1], [0.6, 0.8, 0.1], [0.9, 0.3, 0.4]])
    test_y = np.array([[1], [0], [1]])
    mlp = MultiLayerPerceptron(num_inputs=3, num_hidden=2, hidden_width=4, activation_function=sigmoid, activation_derivated=sigmoid_derivative)

    print(mlp.predict(test_x)[0])
    mlp.fit(train_x=test_x, train_y=test_y, verbose=True, num_epochs=10000)
    print(mlp.predict(test_x)[0])