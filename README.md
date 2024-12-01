# Supervised Models - Examples

The repository shares use cases for supervised models. See the [supervised learning section](https://en.wikipedia.org/wiki/Supervised_learning) for more information.

## Background

The repository is a simple walkthrough to some of the potential use cases for a supervised model.

The dataset used to compare the different use cases is the [Banana Quality Dataset](https://www.kaggle.com/datasets/mrmars1010/banana-quality-dataset).

## Data Pre-Processing

To pre-process the data, first you have to download the dataset from its source. Then you can run the following command:

```
DATA_PATH="the-path-of-your-dataset" python generate_preprocessed_dataset.py
```

You should be able to see 4 new files: `data_test.csv`, `data_train.csv`, `result_test.csv` and `result_train.csv`.

## Use Cases

The repository includes different use cases:

### Neural Network using Back-Propagation

An example implementation using `numpy` to reproduce the functionality of a neural network working with Back Propagation.

#### Pre-requisites

Your dataset files should be already generated. Next you have to install all requirements:

```
pip install -r requirements.txt
```

#### Usage

Run the neural network:

```
python3 NeuralNet.py
```

### Comparison of MLR-F, BP, BP-F

The [model_comparison.ipynb](./model_comparison.ipynb) introduces 2 new use cases:

- A Multi Linear Regression (MLR-F).
- A Neural Network using Back-Propagation (BP-F) with Pytorch.

After running all the commands of the notebook the user can see 3 scatter plots, one for each solution.

## Contribution

Contributions are welcomed to this repository. Feel free to open an issue or a PR with your suggestions.
