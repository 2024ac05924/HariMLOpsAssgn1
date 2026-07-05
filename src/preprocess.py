"""
preprocess.py

This module performs all preprocessing required before model training.

Steps:
1. Load binary heart disease dataset
2. Handle missing values
3. Split features and target
4. Train-Test Split
5. Feature Scaling
6. Save scaler
7. Save processed datasets
"""

from pathlib import Path
import joblib
import pandas as pd
from utils import get_logger
logger = get_logger()
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------
# Project Directories
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

def load_dataset():

    dataset_path = PROCESSED_DIR / "heart_disease_binary.csv"

    df = pd.read_csv(dataset_path)

    return df


# -------------------------------------------------------
# Preprocess
# -------------------------------------------------------

def preprocess_data(test_size=0.2, random_state=42):

    print("=" * 60)
    print("Loading Dataset...")
    logger.info("Loading Dataset...")
    print("=" * 60)

    df = load_dataset()

    print(df.shape)

    # -----------------------------
    # Missing Values
    # -----------------------------

    print("\nHandling Missing Values...")
    logger.info("Handling Missing Values...")
    X = df.drop(columns=["target"])
    y = df["target"]

    imputer = SimpleImputer(strategy="median")

    X = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )

    # -----------------------------
    # Train Test Split
    # -----------------------------

    print("\nSplitting Dataset...")
    logger.info("Splitting Dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")

    # -----------------------------
    # Scaling
    # -----------------------------

    print("\nScaling Features...")
    logger.info("Scaling Features...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # Save scaler
    # -----------------------------

    scaler_path = ARTIFACT_DIR / "scaler.pkl"

    joblib.dump(scaler, scaler_path)

    print(f"\nScaler Saved : {scaler_path}")

    # -----------------------------
    # Save processed datasets
    # -----------------------------

    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        PROCESSED_DIR / "X_train.csv",
        index=False
    )

    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        PROCESSED_DIR / "X_test.csv",
        index=False
    )

    y_train.to_csv(
        PROCESSED_DIR / "y_train.csv",
        index=False
    )

    y_test.to_csv(
        PROCESSED_DIR / "y_test.csv",
        index=False
    )

    print("\nProcessed datasets saved.")
    logger.info("Processed datasets saved.")
    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":

    preprocess_data()