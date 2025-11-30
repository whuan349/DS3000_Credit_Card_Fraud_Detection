"""
Task 4 — Model Evaluation and Comparison
Author: William Huang

Evaluating all trained machine learning models
(Logistic Regression, Random Forest, and XGBoost) 
using the test datasetprepared in Task 2.

1. Load the preprocessed test dataset (X_test, y_test).
2. Load the trained models saved by Task 3.
3. Compute key performance metrics for each model:
    - Precision
    - Recall
    - F1-score
    - AUC-ROC
4. Plot and compare:
    - ROC curves
    - Precision-Recall curves
5. Produce a comparison table summarizing all metrics.
6. (Optional for later): Save plots and results for reporting.
"""

# Import libraries
from train_models import train_all_models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    r2_score,
    mean_squared_error
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Main Evaluation
def evaluate_models():

    # Load Trained Models and Test Data
    results = train_all_models()

    # Trained model objects
    log_reg = results["log_reg"]
    rf = results["random_forest"]
    xgb = results["xgboost"]


    # Test dataset from Task 2 output
    X_test = results["data"]["X_test"]
    y_test = results["data"]["y_test"]

    models = {
            "Logistic Regression": log_reg,
            "Random Forest": rf,
            "XGBoost": xgb
        }

    # Dictionary to store evaluation metrics
    metrics = {}

    for name, model in models.items():
        # Predict labels
        y_pred = model.predict(X_test)
        # Predict fraud probabilities
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Save metrics
        metrics[name] = {
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_prob),
            "Accuracy": accuracy_score(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred),
            "RMSE": rmse
        }

        # Print confusion matrix for each model
        print(f"Confusion Matrix — {name}")
        cm = confusion_matrix(y_test, y_pred)
        print(cm, "\n")

        # Plot confusion matrix heatmap
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    # Convert metrics dictionary to DataFrame
    df = pd.DataFrame(metrics).T
    print("Model Performance Overview:\n")
    print(df)