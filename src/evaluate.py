from pathlib import Path

import joblib

from sklearn.metrics import accuracy_score, classification_report

from data_prep import load_data, split_data


# Get the path to the project's root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Set dataset, model and evaluation results paths
DATA_PATH = BASE_DIR / "data/breast_cancer.csv"
MODEL_PATH = BASE_DIR / "models/random_forest.joblib"
RESULTS_PATH = BASE_DIR / "wyniki.txt"

# Use the same data split parameters as in train.py
TEST_SIZE = 0.2
SEED_STATE = 23


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