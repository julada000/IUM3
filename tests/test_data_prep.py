import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from src.data_prep import load_data, split_data


# Task 5 - Unit tests for data preparation

@pytest.fixture
def sample_data():
    # Prepare a small dataset shared by the tests
    return pd.DataFrame({
        "feature": range(100),
        "target": [0, 1] * 50
    })


def test_load_data_returns_dataframe(tmp_path, sample_data):
    # Save test data in a temporary directory
    path = tmp_path / "test_data.csv"
    sample_data.to_csv(path, index=False)

    # Check that loading preserves the dataset
    df = load_data(path)

    assert isinstance(df, pd.DataFrame)
    assert_frame_equal(df, sample_data)


def test_split_sizes(sample_data):
    # Check the requested test set proportion
    X_train, X_test, y_train, y_test = split_data(
        sample_data,
        test_size=0.2,
        seed=23
    )

    assert len(X_test) == pytest.approx(
        len(sample_data) * 0.2,
        abs=1
    )
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_split_preserves_all_rows(sample_data):
    # Check that every observation belongs to exactly one set
    X_train, X_test, y_train, y_test = split_data(
        sample_data,
        test_size=0.2,
        seed=23
    )

    assert len(X_train) + len(X_test) == len(sample_data)
    assert set(X_train.index).isdisjoint(X_test.index)

    assert (
        set(X_train.index) | set(X_test.index)
        == set(sample_data.index)
    )


def test_split_empty_dataframe():
    # Reject an empty dataset with a predictable error
    empty_data = pd.DataFrame(columns=["feature", "target"])

    with pytest.raises(ValueError, match="empty"):
        split_data(empty_data, test_size=0.2, seed=23)


        