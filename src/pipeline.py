"""
pipeline.py

End-to-End Machine Learning Pipeline

This script executes the complete workflow:

1. Download Dataset
2. Preprocess Dataset
3. Train Models
4. Evaluate Best Model

Run:

python src/pipeline.py
"""

from pathlib import Path
import time

from download_data import download_data
from preprocess import preprocess_data
from train import train_models
from evaluate import evaluate_model
from utils import get_logger

logger = get_logger()

def print_banner(title):
    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def run_pipeline():

    start_time = time.time()

    print_banner("HariMLOps Assignment Pipeline Started")
    logger.info("HariMLOps Assignment Pipeline Started")
    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    print_banner("STEP 1 : DATA DOWNLOAD")
    logger.info("STEP 1 : DATA DOWNLOAD")
    #download_data()
    #Instead downloading everytime, we should check if the dataset already exists
    from pathlib import Path

    dataset_path = Path("data/raw/heart_disease.csv")

    if dataset_path.exists():
        print("Dataset already exists. Skipping download.")
    else:
        download_data()
    # This makes the pipeline idempotent and avoids unnecessary downloads.
    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    print_banner("STEP 2 : DATA PREPROCESSING")
    logger.info("STEP 2 : DATA PREPROCESSING")
    preprocess_data()

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    print_banner("STEP 3 : MODEL TRAINING")
    logger.info("STEP 3 : MODEL TRAINING")

    results = train_models()

    print("\nModel Comparison\n")

    print(results)

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    print_banner("STEP 4 : MODEL EVALUATION")
    logger.info("STEP 4 : MODEL EVALUATION")
    metrics = evaluate_model()

    print("\nEvaluation Metrics\n")

    for metric, value in metrics.items():

        print(f"{metric:15}: {value}")

    end_time = time.time()

    total_time = round(end_time - start_time, 2)

    print_banner("PIPELINE COMPLETED")
    logger.info("PIPELINE COMPLETED")
    print(f"Total Execution Time : {total_time} seconds")

    print("\nArtifacts Generated")

    print("- Processed datasets")

    print("- Trained models")

    print("- Best model")

    print("- Evaluation reports")

    print("- Confusion Matrix")

    print("- ROC Curve")

    print("- MLflow Experiment")


if __name__ == "__main__":

    run_pipeline()