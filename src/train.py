"""
train.py

Train multiple machine learning models,
compare performance,
log experiments using MLflow,
and save the best model.
"""

from pathlib import Path
import joblib
import platform
import pandas as pd
import mlflow
import mlflow.sklearn
from utils import get_logger
logger = get_logger()

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from time import perf_counter
from datetime import datetime
import json
import matplotlib.pyplot as plt

# --------------------------------------------------
# Project Directories
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load Processed Data
# --------------------------------------------------

def load_data():

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# Model Dictionary
# --------------------------------------------------

def get_models():

    return {

        "LogisticRegression":
            LogisticRegression(random_state=42),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),

        "XGBoost":
            XGBClassifier(
                random_state=42,
                eval_metric="logloss"
            )
    }


# --------------------------------------------------
# Train Models
# --------------------------------------------------

def train_models():

    X_train, X_test, y_train, y_test = load_data()

    models = get_models()

    results = []

    mlflow.set_experiment("HeartDiseasePrediction_HariMLOpsAssignment")

    print("=" * 60)
    print("MODEL TRAINING STARTED")
    print("=" * 60)

    for name, model in models.items():

        logger.info(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):
            mlflow.set_tags({

                "Assignment": "HariMLOps",

                "Dataset": "UCI Cleveland Heart Disease",

                "Framework": "scikit-learn",

                "Author": "HariPrasad joshi"

            })
            start = perf_counter()

            model.fit(X_train, y_train)

            training_time = perf_counter() - start

            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                probabilities       
            )

            accuracy = accuracy_score(y_test, predictions)

            precision = precision_score(y_test, predictions)

            recall = recall_score(y_test, predictions)

            f1 = f1_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("training_time_seconds", training_time)

            # Log parameters
            mlflow.log_param("model_name", name)
            for key, value in model.get_params().items():
                mlflow.log_param(key, str(value))
            mlflow.log_param(
                "training_samples",
                len(X_train)
            )

            mlflow.log_param(
                "testing_samples",
                len(X_test)
            )

            mlflow.log_param(
                "features",
                X_train.shape[1]
            )
            # Save model in MLflow
            if name != "XGBoost":
                mlflow.sklearn.log_model(
                    model,
                    name=name
            )
            report = classification_report(
                y_test,
                predictions
            )

            report_path = ARTIFACT_DIR / f"{name}_classification_report.txt"

            with open(report_path, "w") as f:

                f.write(report)

            mlflow.log_artifact(report_path)
            # Save model locally
            model_path = MODEL_DIR / f"{name}.pkl"

            joblib.dump(model, model_path)
            cm = confusion_matrix(y_test, predictions)

            plt.figure(figsize=(5,5))

            plt.imshow(cm, cmap="Blues")

            plt.title(f"{name} Confusion Matrix")

            plt.colorbar()

            plt.xlabel("Predicted")

            plt.ylabel("Actual")

            plt.tight_layout()

            cm_path = ARTIFACT_DIR / f"{name}_confusion_matrix.png"

            plt.savefig(cm_path)

            plt.close()

            mlflow.log_artifact(cm_path)
            results.append({

                "Model": name,
                "Accuracy": round(accuracy,4),
                "Precision": round(precision,4),
                "Recall": round(recall,4),
                "F1 Score": round(f1,4),
                "ROC AUC": round(roc_auc,4),
                "Training Time": round(training_time,4)
            })

            print(f"{name} completed.")

            if name == "RandomForest":

                importance = pd.Series(

                    model.feature_importances_,

                    index=X_train.columns

                ).sort_values(ascending=False)

                plt.figure(figsize=(10,6))

                importance.plot(kind="bar")

                plt.title("Feature Importance")

                plt.tight_layout()

                importance_path = ARTIFACT_DIR / "feature_importance.png"

                plt.savefig(importance_path)

                plt.close()

                mlflow.log_artifact(importance_path)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Accuracy",
        ascending=False
    )

    results_df.to_csv(
        MODEL_DIR / "model_results.csv",
        index=False
    )

    best_model_name = results_df.iloc[0]["Model"]

    best_model = joblib.load(
        MODEL_DIR / f"{best_model_name}.pkl"
    )
    best_hyperparameters = best_model.get_params()
    joblib.dump(
        best_model,
        MODEL_DIR / "best_model.pkl"
    )
    best_model_file = "best_model.pkl"
    feature_scaler = "StandardScaler"
    best_metrics = results_df.iloc[0].to_dict()

    metadata = {

        "model_name": best_model_name,

        "dataset": "UCI Cleveland Heart Disease",

        "framework": "scikit-learn",

        "python_version": platform.python_version(),

        "training_samples": len(X_train),

        "testing_samples": len(X_test),

        "features": X_train.shape[1],

        "feature_scaler": feature_scaler,

        "best_model_file": best_model_file,

        "accuracy": float(best_metrics["Accuracy"]),

        "precision": float(best_metrics["Precision"]),

        "recall": float(best_metrics["Recall"]),

        "f1_score": float(best_metrics["F1 Score"]),

        "roc_auc": float(best_metrics["ROC AUC"]),

        "training_time_seconds": float(best_metrics["Training Time"]),

        "hyperparameters": best_hyperparameters,

        "generated_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    metadata_path = ARTIFACT_DIR / "model_metadata.json"

    with open(metadata_path, "w") as file:

        json.dump(metadata, file, indent=4)
    mlflow.log_artifact(str(metadata_path))
    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(results_df)

    print("\nBest Model :", best_model_name)

    logger.info("\nTraining Completed Successfully.")
    mlflow.log_artifact(metadata_path)
    return results_df


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    train_models()