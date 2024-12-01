import math
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

DATA_TEST_PATH = "data_test.csv"
DATA_TRAIN_PATH = "data_train.csv"
HIDDEN_LAYERS = [32, 24, 12]
LEARNING_RATE = 0.01
MOMENTUM = 0.85
NUM_OF_EPOCHS = 1000
OUTPUT_LAYER = [1]
PLOT_FIGURE_SIZE = (8, 6)
RESULTS_TEST_PATH = "result_test.csv"
RESULTS_TRAIN_PATH = "result_train.csv"
VALIDATION_RATIO = 0.15


# activation methods (relu + relu_derivative)
def relu(x: "np.ndarray") -> "np.ndarray":
    return np.maximum(0, x)


def relu_derivative(x: "np.ndarray") -> "np.ndarray":
    return np.where(x > 0, 1, 0)


def sigmoid(x: "np.ndarray") -> "np.ndarray":
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: "np.ndarray") -> "np.ndarray":
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    exp_pos = np.exp(x)
    exp_neg = np.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def tanh_derivative(x):
    return 1 - tanh(x) ** 2


# method to plot the evolution of the neural network
def plot(
    test_error: "float",
    errors: "np.ndarray",
    figsize: "tuple[int, int]" = PLOT_FIGURE_SIZE,
) -> None:
    _num_of_epochs = range(1, len(errors[0]) + 1)

    plt.figure(figsize=figsize)
    plt.plot(_num_of_epochs, errors[0], label="Training Error", marker="o")
    plt.plot(_num_of_epochs, errors[1], label="Validation Error", marker="x")

    # Add labels, title, legend, and grid
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.title(
        f"Training and Validation Error over Epochs - Test error: {test_error:.6f}"
    )
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


class NeuralNet:
    def __init__(
        self,
        total_layers: "int",
        units_per_layer: "list[int]",
        learning_rate: "float",
        momentum: "float",
        fact: "Callable[...]",
        fact_derivative: "Callable[...]",
        validation_percentage: "float",
        num_of_epochs: "int",
    ) -> None:
        self.L = total_layers
        self.n = units_per_layer
        # the total layers should be equal to the length of units per layer
        assert self.L == len(
            self.n
        ), "total layers should be equivalent to the length of units_per_layer"

        # initialize learning rate, activation function, validation percentage,
        # momentum and number of epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.fact = fact
        self.fact_derivative = fact_derivative
        self.validation_percentage = validation_percentage
        self.num_of_epochs = num_of_epochs

        # initialize loss epochs cache
        self._cache_loss_epochs: "list[list[float]]" = []

        # initialize all neurons for all given layers
        self.xi: "list[np.ndarray]" = []
        for layer in range(self.L):
            self.xi.append(np.zeros(self.n[layer]))

        # initialize randomly all weights and thresholds for all given layers
        self.w: "list[np.ndarray]" = []
        self.theta: "list[np.ndarray]" = []
        for layer in range(1, self.L):
            self.w.append(np.random.randn(self.n[layer], self.n[layer - 1]))
            self.theta.append(np.random.randn(self.n[layer], 1))

        self.d_w_prev = None
        self.d_theta_prev = None

    def _forward(self, x: "np.ndarray") -> "np.ndarray":
        """
        performs feed forward propagation for a given x train data array
        """
        self.xi[0] = x.reshape(-1, 1)
        for layer in range(1, self.L):
            h = np.dot(self.w[layer - 1], self.xi[layer - 1]) - self.theta[layer - 1]
            self.xi[layer] = self.fact(h)
        return self.xi[-1]

    def _backward(self, y: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
        """
        performs back propagation for a given y array with the result
        """
        delta = (self.xi[-1] - y) * self.fact_derivative(self.xi[-1])
        d_w = [None] * self.L
        d_theta = [None] * self.L

        for layer in reversed(range(self.L)):
            if self.d_w_prev is None:
                d_w[layer] = -self.learning_rate * (delta @ self.xi[layer - 1].T)
                d_theta[layer] = self.learning_rate * delta
            else:
                d_w[layer] = (
                    -self.learning_rate * (delta @ self.xi[layer - 1].T)
                    + self.momentum * self.d_w_prev[layer]
                )
                d_theta[layer] = (
                    self.learning_rate * delta
                    + self.momentum * self.d_theta_prev[layer]
                )

            if layer > 1:
                delta = (self.w[layer - 1].T @ delta) * self.fact_derivative(
                    self.xi[layer - 1]
                )

        self.d_w_prev = d_w
        self.d_theta_prev = d_theta
        return d_w, d_theta

    def _update_weights(
        self,
        d_w: "list[np.ndarray]",
        d_theta: "list[np.ndarray]",
    ) -> None:

        for layer in range(1, self.L):
            self.w[layer - 1] += d_w[layer]
            self.theta[layer - 1] += d_theta[layer]

    def _compute_mean_squared_error(self, X: "np.ndarray", y: "np.ndarray") -> "float":
        total_records, _ = X.shape
        predictions = []
        for i in range(total_records):
            pred = self._forward(X[i])
            predictions.append(pred[0][0])
        return mean_squared_error(predictions, y)

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        predictions = []
        total_records, _ = X.shape
        for i in range(total_records):
            predictions.append(self._forward(X[i])[0][0])
        return np.array(predictions)

    def loss_epochs(self) -> "np.ndarray":
        return self._cache_loss_epochs

    def _split_data(
        self, data: "np.ndarray", total_records: "int"
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        splits the data into validation and training according to the give
        validation percentage
        """
        _val_splitter = int(self.validation_percentage * total_records)

        return data[_val_splitter:], data[:_val_splitter]

    def fit(
        self,
        X: "np.ndarray",
        y: "np.ndarray",
    ) -> None:
        _train_errors: "list[float]" = []
        _val_errors: "list[float]" = []
        # Training the network with mini-batch stochastic gradient descent
        original_total_records, _ = X.shape
        # Split output train into train and validation
        y_train, y_val = self._split_data(y, original_total_records)
        X_train, data_val = self._split_data(X, original_total_records)

        # empty cache loss epochs
        self._cache_loss_epochs = []
        for epoch in range(self.num_of_epochs):
            for _ in range(X_train.shape[0]):
                # select a random item from the records
                m = np.random.randint(X_train.shape[0])

                # Run the forward propagation
                self._forward(X_train[m])

                # Get the weight and threshold changes after backward propagation
                d_w, d_theta = self._backward(y_train[m])

                # Adjust weight according to results of training
                self._update_weights(d_w, d_theta)

            # After an epoch is finished calculate the error for training and validation sets
            # _train_error = mean_squared_error(X_train, y_train)
            _train_error = self._compute_mean_squared_error(X_train, y_train)
            _val_error = (
                0
                if self.validation_percentage == 0
                else self._compute_mean_squared_error(data_val, y_val)
            )
            # _val_error = mean_squared_error(data_val, y_val)
            if epoch % 10 == 0 or epoch == self.num_of_epochs - 1:
                print(
                    f"Epoch {epoch}, Train Error: {_train_error:.6f}, Val Error: {_val_error:.6f}"
                )

            # Populate the train and validation error lists
            _train_errors.append(_train_error)
            _val_errors.append(_val_error)

        self._cache_loss_epochs = np.array([_train_errors, _val_errors])


if __name__ == "__main__":
    X_train = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",")
    y_train = np.genfromtxt(RESULTS_TRAIN_PATH, delimiter=",")
    X_test = np.genfromtxt(DATA_TEST_PATH, delimiter=",")
    y_test = np.genfromtxt(RESULTS_TEST_PATH, delimiter=",")

    # Get the total number of features
    _, total_features = X_train.shape

    # Initialize the neural network
    total_layers = [total_features] + HIDDEN_LAYERS + OUTPUT_LAYER
    neuronet = NeuralNet(
        total_layers=len(total_layers),
        units_per_layer=total_layers,
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        fact=tanh,
        fact_derivative=tanh_derivative,
        validation_percentage=VALIDATION_RATIO,
        num_of_epochs=NUM_OF_EPOCHS,
    )

    # Train the neural network
    neuronet.fit(
        X_train,
        y_train,
    )

    # After training we are ready to test
    predictions = neuronet.predict(X_test)

    # Evaluate test performance
    test_error = np.mean((predictions - y_test) ** 2)

    # Get the neural network BP mean square, mean absolute and mean absolute percentage error
    nn_mean_square_error = mean_squared_error(y_test, predictions)
    nn_mean_absolute_error = mean_absolute_error(y_test, predictions)
    nn_mean_absolute_percentage_error = mean_absolute_percentage_error(
        y_test, predictions
    )

    print(f"Neural Network with BP Mean Square Error:: {nn_mean_square_error:.6f}")
    print(f"Neural Network with BP Mean Absolute Error:: {nn_mean_absolute_error:.6f}")
    print(
        f"Neural Network with BP Mean Absolute Percentage Error:: {nn_mean_absolute_percentage_error:.6f}"
    )

    # Plot the results
    plot(test_error, neuronet.loss_epochs())
