import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# Task 1 - Data loading and splitting

def load_data(path):
    # Read the dataset without modifying the source file
    df = pd.read_csv(path)

    # Check that the dataset contains observations, features and labels
    if df.empty:
        raise ValueError("The dataset cannot be empty.")

    if "target" not in df.columns or len(df.columns) < 2:
        raise ValueError("The dataset must contain features and target.")

    if not all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes
    ):
        raise ValueError("The dataset must contain numeric values.")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(
            "The dataset cannot contain missing or infinite values."
        )

    return df


def split_data(df, test_size, seed):
    # Validate the input before separating features and labels
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The dataset cannot be empty.")

    if "target" not in df.columns or len(df.columns) < 2:
        raise ValueError("The dataset must contain features and target.")

    if df.isnull().any().any():
        raise ValueError("The dataset cannot contain missing values.")

    if not 0 < test_size < 1:
        raise ValueError("TEST_SIZE must be between 0 and 1.")

    # Separate input features from class labels
    X = df.drop(columns=["target"])
    y = df["target"]

    # Preserve class proportions and use a reproducible data split
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )