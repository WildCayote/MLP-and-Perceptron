
import numpy as np


class Perceptron(object):
    def __init__(self, lr=0.01, n_iter=10, seed=42, weights_random=True):
        self.lr = lr
        self.n_iter = n_iter
        self.seed = seed
        self.weights_random = weights_random
        self.boundaries = []  # To store decision boundaries

    def weighted_sum(self, X):
        return self.w_[0] + np.dot(X, self.w_[1:])

    def heaviside(self, X):  # predict
        return np.where(self.weighted_sum(X) > 0, 1, 0)

    def fit(self, X, y):
        np.random.seed(self.seed)

        if self.weights_random:
            self.w_ = np.random.rand(X.shape[1] + 1) * 0.01
        else:
            self.w_ = np.zeros(X.shape[1] + 1)

        self.errors = []

        for epoch in range(self.n_iter):
            error = 0
            for x, y_ in zip(X, y):
                y_pred = self.heaviside(x)
                delta = self.lr * (y_ - y_pred)

                self.w_[1:] = self.w_[1:] + delta * x
                self.w_[0] = self.w_[0] + delta  # Update bias term

                error += int(delta != 0.0)

            self.errors.append(error)

            # Record the boundary weights after each epoch
            self.boundaries.append(self.w_.copy())

        return self


if __name__ == '__main__':
    ...
    