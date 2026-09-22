import os
import json
import mlflow
import mlflow.sklearn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.config import (
    DATA_PATH,
    ARTIFACT_DIR,
    EXPERIMENT_NAME,
    TRACKING_URI,
    TARGET_COLUMN,
    RANDOM_STATE
)


def load_data():
    """
    Load the processed Heart Disease dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def evaluate_model(model, X_test, y_test):
    """
    Calculate classification metrics.
    """

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    return predictions, metrics


def save_confusion_matrix(y_test, predictions, model_name):
    """
    Save the confusion matrix as an image.
    """

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    output_path = ARTIFACT_DIR / f"{model_name}_confusion_matrix.png"

    plt.savefig(output_path)
    plt.close()

    return output_path


def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Train a model and log everything using MLflow.
    """

    with mlflow.start_run(run_name=model_name) as run:

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate
        predictions, metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        # Log model name
        mlflow.log_param("model_name", model_name)

        # Log model parameters
        model_params = model.get_params()

        for parameter, value in model_params.items():

            if value is not None and isinstance(
                value,
                (str, int, float, bool)
            ):
                mlflow.log_param(
                    parameter,
                    value
                )

        # Log metrics
        mlflow.log_metrics(metrics)

        # Save and log confusion matrix
        confusion_matrix_path = save_confusion_matrix(
            y_test,
            predictions,
            model_name
        )

        mlflow.log_artifact(
            str(confusion_matrix_path)
        )

        # Save classification report
        report = classification_report(
            y_test,
            predictions,
            zero_division=0
        )

        report_path = ARTIFACT_DIR / f"{model_name}_report.txt"

        with open(report_path, "w") as file:
            file.write(report)

        mlflow.log_artifact(str(report_path))

        # Log trained model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        print("\nModel:", model_name)
        print("Run ID:", run.info.run_id)
        print("Metrics:", metrics)

        return run.info.run_id, metrics


def main():

    # Configure MLflow
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load dataset
    X, y = load_data()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Define models
    models = {

        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ]),

        "Decision_Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=RANDOM_STATE
        )
    }

    results = {}

    # Train all models
    for model_name, model in models.items():

        run_id, metrics = train_and_log_model(
            model_name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results[model_name] = {
            "run_id": run_id,
            "metrics": metrics
        }

    # Save summary
    summary_path = ARTIFACT_DIR / "model_comparison.json"

    with open(summary_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\nAll models trained successfully.")
    print("Comparison summary saved at:", summary_path)


if __name__ == "__main__":
    main()
    