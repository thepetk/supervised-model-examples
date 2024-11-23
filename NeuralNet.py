from typing import Any
import numpy as np

DATA_TEST_PATH = "data_test.csv"
DATA_TRAIN_PATH = "data_train.csv"
HIDDEN_LAYERS = [40, 20, 5]
LEARNING_RATE = 0.01
NUM_OF_EPOCHS = 1000
OUTPUT_LAYER = [1]
RESULTS_TEST_PATH = "result_test.csv"
RESULTS_TRAIN_PATH = "result_train.csv"
VALIDATION_RATIO = 0.15


# activation methods
def fact(z):
    return 1 / (1 + np.exp(-z))


def fact_derivative(z):
    return z * (1 - z)


class NeuralNet:
    def __init__(self, layers: "list[int]") -> None:
        self.L = len(layers)
        self.n = layers.copy()

        # initialize all neurons for all given layers
        self.xi: "list[np.ndarray]" = []
        for layer in range(self.L):
            self.xi.append(np.zeros(self.n[layer]))

        # initialize all weights and thresholds for all given layers
        self.w: "list[np.ndarray]" = []
        self.theta: "list[np.ndarray]" = []
        for layer in range(1, self.L):
            self.w.append(np.zeros((self.n[layer], self.n[layer - 1])))
            self.theta.append(np.zeros((self.n[layer], 1)))

    def forward(self, x: "np.ndarray") -> "np.ndarray":
        self.xi[0] = x.reshape(-1, 1)
        for layer in range(1, self.L):
            z = self.w[layer - 1] @ self.xi[layer - 1] + self.theta[layer - 1]
            self.xi[layer] = fact(z)
        return self.xi[-1]

    def backward(self, y: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
        residual = self.xi[-1] - y
        d_w = [None] * self.L
        d_theta = [None] * self.L

        for layer in reversed(range(self.L)):
            d_w[layer] = residual @ self.xi[layer - 1].T
            d_theta[layer] = residual
            if layer > 1:
                residual = (
                    self.w[layer - 1].T @ residual * fact_derivative(self.xi[layer - 1])
                )

        return d_w, d_theta

    def update_weights(
        self,
        d_w: "list[np.ndarray]",
        d_theta: "list[np.ndarray]",
        learning_rate: "float",
    ) -> None:
        for layer in range(1, self.L):
            self.w[layer - 1] -= learning_rate * d_w[layer]
            self.theta[layer - 1] -= learning_rate * d_theta[layer]

    def compute_mean_square_error(self, data: "np.ndarray", y: "np.ndarray") -> Any:
        error = 0.0
        total_records, _ = data.shape
        for i in range(total_records):
            pred = self.forward(data[i])
            error += np.sum((pred - y[i]) ** 2)
        return error / total_records

    def predict(self, data: "np.ndarray") -> "np.ndarray":
        predictions = []
        total_records, _ = data.shape
        for i in range(total_records):
            predictions.append(self.forward(data[i]))
        return np.array(predictions)

    def train(
        self,
        data_train: "np.ndarray",
        y_train: "np.ndarray",
        data_val: "np.ndarray",
        y_val: "np.ndarray",
        epochs: "int",
        learning_rate: "float",
    ) -> None:
        # Training the network with mini-batch stochastic gradient descent
        total_records, _ = data_train.shape
        for epoch in range(epochs):
            for _ in range(total_records):
                # select a random item from the records
                m = np.random.randint(total_records)

                # Run the forward propagation
                self.forward(data_train[m])

                # Get the weight gradients after backward propagation
                d_w, d_theta = self.backward(y_train[m])

                # Adjust weight according to results of training
                self.update_weights(d_w, d_theta, learning_rate)

            # After an epoch is finished calculate the error for training and validation sets
            train_error = self.compute_mean_square_error(data_train, y_train)
            val_error = self.compute_mean_square_error(data_val, y_val)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch}, Train Error: {train_error:.6f}, Val Error: {val_error:.6f}"
                )


if __name__ == "__main__":
    data_train = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",")
    y_train = np.genfromtxt(RESULTS_TRAIN_PATH, delimiter=",")
    test_data = np.genfromtxt(DATA_TEST_PATH, delimiter=",")
    test_y = np.genfromtxt(RESULTS_TEST_PATH, delimiter=",")

    # Get the total number of records
    total_records, _ = data_train.shape

    # Define a split between validation and training
    val_splitter = int(VALIDATION_RATIO * total_records)

    # Split data train into train and validation
    data_train = data_train[val_splitter:]
    data_val = data_train[:val_splitter]

    # Split output train into train and validation
    y_train = y_train[val_splitter:]
    y_val = y_train[:val_splitter]

    # Get the total number of features
    _, total_features = data_train.shape

    # Initialize the neural network
    total_layers = [total_features] + HIDDEN_LAYERS + OUTPUT_LAYER
    neuronet = NeuralNet(layers=total_layers)

    print(y_train[0])
    # Train the neural network
    neuronet.train(
        data_train,
        y_train,
        data_val,
        y_val,
        epochs=NUM_OF_EPOCHS,
        learning_rate=LEARNING_RATE,
    )

    # After training we are ready to test
    predictions = neuronet.predict(test_data)

    # Evaluate test performance
    test_error = np.mean((predictions - test_y) ** 2)
    print(f"Test Error (MSE): {test_error:.6f}")
