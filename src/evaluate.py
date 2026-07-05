"""
evaluate.py

Evaluates the trained models and generates:

1. Accuracy
2. Precision
3. Recall
4. F1 Score
5. ROC AUC
6. Classification Report
7. Confusion Matrix
8. ROC Curve

Outputs are saved under:

artifacts/evaluation/
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_logger

logger = get_logger()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# -----------------------------------------------------
# Project Directories
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------
# Load Data
# -----------------------------------------------------

def load_test_data():

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_test = (
        pd.read_csv(DATA_DIR / "y_test.csv")
        .values
        .ravel()
    )

    return X_test, y_test


# -----------------------------------------------------
# Load Best Model
# -----------------------------------------------------

def load_best_model():

    model = joblib.load(
        MODEL_DIR / "best_model.pkl"
    )

    return model


# -----------------------------------------------------
# Evaluate Model
# -----------------------------------------------------

def evaluate_model():

    print("=" * 60)
    print("MODEL EVALUATION")
    logger.info("MODEL EVALUATION")
    print("=" * 60)

    X_test, y_test = load_test_data()

    model = load_best_model()

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    metrics = {

        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4)

    }

    print("\nEvaluation Metrics\n")
    logger.info("Evaluation Metrics")
    for key, value in metrics.items():

        print(f"{key:12}: {value}")
        logger.info(f"{key:12}: {value}")
    # -------------------------------------------------
    # Save Metrics
    # -------------------------------------------------

    with open(
        OUTPUT_DIR / "metrics.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------

    report = classification_report(
        y_test,
        predictions
    )

    print("\nClassification Report\n")

    print(report)

    with open(
        OUTPUT_DIR /
        "classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # -------------------------------------------------
    # ROC Curve
    # -------------------------------------------------

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.title("ROC Curve")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR /
        "roc_curve.png",
        dpi=300
    )

    plt.close()

    print("\nEvaluation artifacts saved.")
    logger.info("Evaluation artifacts saved.")
    print(OUTPUT_DIR)

    return metrics


# -----------------------------------------------------
# Main
# -----------------------------------------------------

if __name__ == "__main__":

    evaluate_model()