from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from NeuralNet import (
    FACT,
    FACT_DERIVATIVE,
    NeuralNet,
    MOMENTUM,
    HIDDEN_LAYERS,
    LEARNING_RATE,
    NUM_OF_EPOCHS,
    OUTPUT_LAYER,
    VALIDATION_RATIO,
)


DATA_TEST_PATH = "data_test.csv"
DATA_TRAIN_PATH = "data_train.csv"
PLOT_FIGURE_SIZE = (8, 6)
RESULTS_TEST_PATH = "result_test.csv"
RESULTS_TRAIN_PATH = "result_train.csv"


@dataclass
class CrossValidationParameterSet:
    name: "str"
    learning_rate: "float"
    momentum: "float"
    num_of_epochs: "int"


K = 3
K_FOLD_LEVELS = [
    # Best case
    CrossValidationParameterSet(
        name="best case (1)",
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        num_of_epochs=NUM_OF_EPOCHS,
    ),
    CrossValidationParameterSet(
        name="lower momentum (2)",
        learning_rate=LEARNING_RATE,
        momentum=0.85,
        num_of_epochs=NUM_OF_EPOCHS,
    ),
    CrossValidationParameterSet(
        name="lower num of epochs (3)",
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        num_of_epochs=500,
    ),
    CrossValidationParameterSet(
        name="increased learning rate (4)",
        learning_rate=0.05,
        momentum=MOMENTUM,
        num_of_epochs=NUM_OF_EPOCHS,
    ),
]
if __name__ == "__main__":
    total_mse = 0
    total_mae = 0
    total_mape = 0
    total_iterations = 0
    X_train = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",")
    y_train = np.genfromtxt(RESULTS_TRAIN_PATH, delimiter=",")
    X_test = np.genfromtxt(DATA_TEST_PATH, delimiter=",")
    y_test = np.genfromtxt(RESULTS_TEST_PATH, delimiter=",")

    _, total_features = X_train.shape
    total_layers = [total_features] + HIDDEN_LAYERS + OUTPUT_LAYER

    for case in K_FOLD_LEVELS:
        print(f"Running Case: {case.name}...")
        for k in range(K):
            print(f"K={k}")
            neuronet = NeuralNet(
                total_layers=len(total_layers),
                units_per_layer=total_layers,
                learning_rate=case.learning_rate,
                momentum=case.momentum,
                fact=FACT,
                fact_derivative=FACT_DERIVATIVE,
                validation_percentage=VALIDATION_RATIO,
                num_of_epochs=case.num_of_epochs,
                verbose=True,
            )

            neuronet.fit(
                X_train,
                y_train,
            )
            predictions = neuronet.predict(X_test)

            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            total_mse += mse
            total_mae += mae
            total_mape += mape
            total_iterations += 1

            print(f"MSE:: {mse:.6f}")
            print(f"MAE:: {mae:.6f}")
            print(f"MAPE:: {mape:.6f}")

    print(f"Mean MAE:: {total_mae/total_iterations:.6f}")
    print(f"Mean MSE:: {total_mse/total_iterations:.6f}")
    print(f"Mean MAPE:: {total_mape/total_iterations:.6f}")
