# OpenSpec.md

# Heart Disease Prediction using MLOps

## Project Objective

Develop an end-to-end MLOps pipeline to predict whether a patient has heart disease using the UCI Heart Disease dataset.

This project is submitted as part of the MLOps assignment.

---

# Problem Statement

The original UCI Heart Disease dataset contains target values:

0 = No Heart Disease

1 = Low Severity

2 = Medium Severity

3 = High Severity

4 = Very High Severity

For this assignment:

Target = 0 → No Heart Disease

Target = 1,2,3,4 → Heart Disease

Hence, convert the target into Binary Classification.

---

# Dataset

Source

https://archive.ics.uci.edu/ml/datasets/heart+disease

Dataset used

processed.cleveland.data

---

# Folder Structure

HariMLOpsAssgn1

data/
    raw/
    processed/

models/

notebooks/

src/

tests/

docker/

k8s/

monitoring/

reports/

screenshots/

mlruns/

---

# Models

The following models should be implemented.

1. Logistic Regression

2. Random Forest

3. XGBoost

The best performing model should be selected.

---

# Notebook Guidelines

Every notebook should contain

• Title

• Objective

• Dataset Description

• Import Libraries

• Data Loading

• Data Cleaning

• Exploratory Data Analysis

• Observations

• Conclusion

---

# Coding Guidelines

Use

- Functions

- Comments

- Docstrings

- Meaningful variable names

- Modular code

Avoid

- Hard coded values

- Duplicate code

---

# Data Cleaning

Handle missing values represented by ?

Convert columns to correct data types

Convert target

0 → 0

1,2,3,4 → 1

Remove duplicates if any.

---

# Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

ROC AUC

Confusion Matrix

---

# MLflow

Track

Parameters

Metrics

Artifacts

Model

---

# FastAPI

Create endpoint

/predict

Return JSON prediction.

---

# Docker

Containerize FastAPI application.

---

# GitHub Actions

Automatically run tests.

---

# Kubernetes

Deploy FastAPI container.

---

# Monitoring

Prometheus

Grafana

---

# Report

Include

Problem Statement

Methodology

EDA

Model Comparison

Results

Deployment

Conclusion

---

# Development Workflow

This project will be developed incrementally.

Each notebook should follow the structure below:

1. Title
2. Objective
3. Required Libraries
4. Data Loading
5. Data Inspection
6. Data Cleaning
7. Exploratory Data Analysis
8. Observations
9. Conclusion

Guidelines:

- Use separate cells for each logical task.
- Do not combine multiple analyses into one cell.
- Add a short observation after every visualization.
- Keep explanations concise and student-friendly.
- Write code that is modular and easy to understand.
- Verify the output after each major section before proceeding to the next.