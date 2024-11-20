import os
import logging
import pandas as pd

from typing import Literal
from sklearn.preprocessing import OneHotEncoder

AVAILABLE_NULL_POLICIES = ("all", "any")
DATA_PATH = os.getenv("DATA_PATH", "banana_quality_dataset.csv")
DEFAULT_NULL_POLICY = "any"


class CSVImporter:
    def read_raw_data(self, csv_file_path: str) -> pd.DataFrame:
        """
        reads a csv file using the pandas library
        """
        return pd.read_csv(csv_file_path, decimal=".")


class PreProcessor:
    def __init__(
        self,
        data: "pd.DataFrame",
        null_policy: 'Literal["any", "all"]' = DEFAULT_NULL_POLICY,
        remove_columns: "list[str] | None" = None,
    ) -> None:
        self._raw_data = data
        self.data = data
        self.null_policy = self._get_null_policy(null_policy)
        self.remove_columns = remove_columns
        self.one_hot_encoder = OneHotEncoder(sparse_output=True)

    def _get_null_policy(
        self, null_policy: 'Literal["any", "all"]'
    ) -> 'Literal["any", "all"]':
        """
        Ensures that the given null policy is in the list of available null policies
        """
        if null_policy not in AVAILABLE_NULL_POLICIES:
            logging.warning(
                f"given null policy not in {AVAILABLE_NULL_POLICIES}. Defaulting to {DEFAULT_NULL_POLICY}"
            )
            return DEFAULT_NULL_POLICY
        return null_policy

    def _apply_null_policy(self) -> "pd.DataFrame":
        """
        removes all null-na rows found using the given null policy
        """
        return self.data.dropna(axis=0, how=self.null_policy)

    def _clean_excluded_columns(self) -> "pd.DataFrame":
        """
        cleans data from not needed columns
        """
        return (
            self.data
            if not self.remove_columns
            else self.data.drop(columns=self.remove_columns)
        )

    def _get_caterogical_columns(self) -> "list[str]":
        """
        fetches all categorical columns inside the given data
        """
        return self.data.select_dtypes(include=["object"]).columns.tolist()

    def _apply_one_hot_encoding(self) -> "pd.DataFrame":
        """
        applies one hot encoding on the data for all categorical columns
        """
        _categorical_cols = self._get_caterogical_columns()
        _array = self.one_hot_encoder.fit_transform(self.data[_categorical_cols])
        return pd.DataFrame(
            _array,
            columns=self.one_hot_encoder.get_feature_names_out(_categorical_cols),
        )

    def _clean_all_categorical_columns(self) -> "pd.DataFrame":
        """
        keeps only numerical columns in data
        """
        return self.data.drop(columns=self._get_caterogical_columns())

    def _normalize(self) -> "pd.DataFrame":
        """
        normalizes the data by subtracting the mean and divide with standard deviation per column
        """
        return (self.data - self.data.mean()) / self.data.std()

    def _concatinate(self, concat_data_frame: "pd.DataFrame") -> "pd.DataFrame":
        """concatinates data with a given dataframe"""
        return pd.concat([self.data, concat_data_frame], axis=1)

    def preprocess(self) -> "tuple[list[float, pd.DataFrame]]":
        """
        applies the data preprocessing. Returns a tuple with the output and the preprocessed dataset
        """
        # handle any null values
        self.data = self._apply_null_policy()

        # remove the unwanted columns
        self.data = self._clean_excluded_columns()

        # apply the one hot encoding on the categorical columns
        _one_hot_encoding_df = self._apply_one_hot_encoding()

        # remove the categorical columns after one hot encoding
        self.data = self._clean_all_categorical_columns()

        # normalize the dataset
        self.data = self._normalize()

        # concatinate with one hot encoder result
        self.data = self._concatinate(_one_hot_encoding_df)

        # prepare the output
        output = self.data["quality_score"]

        # prepare the features
        features = self.data.drop(columns=["quality_score"])

        return output, features


if __name__ == "__main__":
    # Import the csv file
    csv_importer = CSVImporter()
    raw_data = csv_importer.read_raw_data(DATA_PATH)

    # Create a preprocessor instance for the data preprocessing
    data_manager = PreProcessor(
        data=raw_data,
        null_policy="any",
        remove_columns=[
            "sample_id",
            "quality_category",
            "ripeness_category",
            "harvest_date",
        ],
    )
    output, dataset = data_manager.preprocess()

    # Split into two sets (train/validation and test)
    train_dataset = dataset.sample(frac=0.8, random_state=45)
    test_dataset = dataset.drop(train_dataset.index)

    # Generate the new csv files
    train_dataset.to_csv("data_train.csv", index=False)
    test_dataset.to_csv("data_test.csv", index=False)
