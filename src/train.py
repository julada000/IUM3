from pathlib import Path

import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_data, split_data

import os
from dotenv import load_dotenv


# Load configuration from the project's root directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Read dataset and model paths
DATA_PATH = BASE_DIR / os.environ["DATA_PATH"]
MODEL_PATH = BASE_DIR / os.environ["MODEL_PATH"]

# Read data split and model parameters
TEST_SIZE = float(os.environ["TEST_SIZE"])
SEED_STATE = int(os.environ["RANDOM_SEED"])
N_ESTIMATORS = int(os.environ["N_ESTIMATORS"])


# Task 2 - Model training and saving

def main():
    # Load the dataset and prepare the training and test sets
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=TEST_SIZE,
        seed=SEED_STATE
    )

    # Create a pipeline with scaling and Random Forest
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=SEED_STATE
        ))
    ])

    # Train the whole pipeline on the training set
    model.fit(X_train, y_train)

    # Create the model directory if it does not already exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save the fitted scaler and classifier together
    joblib.dump(model, MODEL_PATH)

    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)
    print("Model training completed successfully!")
    print(f"Model saved to {MODEL_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
