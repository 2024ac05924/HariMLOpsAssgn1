from pathlib import Path
import pandas as pd

# URL for the Cleveland heart disease dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Meaningful column names for the dataset
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_data():
    """Download the heart disease dataset and save it as a CSV file."""
    # Create the output folder if it does not exist
    output_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "heart_disease.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the dataset from the UCI repository
    # Treat '?' as missing values so the data can be cleaned later
    df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES, na_values="?")

    # Save the dataset to the required raw data location
    df.to_csv(output_path, index=False)

    # Print basic dataset information for verification
    print(f"Dataset shape: {df.shape}")
    print("\nFirst five rows:")
    print(df.head())


if __name__ == "__main__":
    download_data()
