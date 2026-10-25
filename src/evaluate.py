from pathlib import Path

import joblib

from sklearn.metrics import accuracy_score, classification_report

from data_prep import load_data, split_data

import os
from dotenv import load_dotenv


# Load configuration from the project's root directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Read dataset, model and evaluation results paths
DATA_PATH = BASE_DIR / os.environ["DATA_PATH"]
MODEL_PATH = BASE_DIR / os.environ["MODEL_PATH"]
RESULTS_PATH = BASE_DIR / os.environ["RESULTS_PATH"]

# Read the same data split parameters as in train.py
TEST_SIZE = float(os.environ["TEST_SIZE"])
SEED_STATE = int(os.environ["RANDOM_SEED"])


# Task 3 - Independent model evaluation

def main():
    # Load the dataset and reproduce the same data split
    df = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=TEST_SIZE,
        seed=SEED_STATE
    )

    # Load the saved pipeline without training it again
    model = joblib.load(MODEL_PATH)

    # Predict class labels for the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and the classification report
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )

    # Prepare the evaluation results
    results = (
        f"Accuracy: {accuracy:.4f}\n\n"
        f"Classification Report:\n{report}"
    )

    # Save the results as a UTF-8 text file
    RESULTS_PATH.write_text(results, encoding="utf-8")

    print(results)
    print(f"Evaluation results saved to {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
