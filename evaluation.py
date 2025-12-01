"""
Task 4 — Model Evaluation and Comparison
Author: William Huang

Evaluating all trained machine learning models
(Logistic Regression, Random Forest, and XGBoost) 
using the test datasetprepared in Task 2.

1. Load the preprocessed test dataset (X_test, y_test).
2. Load the trained models saved by Task 3.
3. Compute key performance metrics for each model:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - AUC-ROC
    - R2 Score
    - RMSE
4. Produce a comparison table summarizing all metrics.
5. Plot and compare:
    - ROC curves
    - Precision-Recall curves
6. Determine Random Forest feature importance on the PCA features.
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


# Main Evaluation Function
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
        # Results of confusion matrices help to visualize model performance in terms of true/false positives and negatives
        # This is crucial for understanding how well the model distinguishes between classes, which is a main goal of our project


    # Convert metrics dictionary to DataFrame
    df = pd.DataFrame(metrics).T
    print("Model Performance Overview:\n")
    print(df)



    # ROC Curves
    # Results from this plot help to visualize model performance across different thresholds
    # Higher AUC indicates better model performance, which this plot illustrates extremely well
    print("Generating ROC curves...\n")
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()



    # Precision-Recall Curves
    # Results from this plot help to understand model performance on imbalanced datasets
    print("Generating Precision–Recall curves...\n")
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(recall, precision, label=name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Random Forest feature importances
    # This demonstrates the importance of PCA features used in the Random Forest model
    print("Computing Random Forest feature importance...\n")
    importances = rf.feature_importances_
    # Handle numpy arrays and pandas index for columns
    try:
        feature_names = X_test.columns
    except Exception:
        # If X_test is a numpy array, create generic feature names
        feature_names = [f"f{i}" for i in range(X_test.shape[1])]

    sorted_index = np.argsort(importances)

    plt.figure(figsize=(8, max(4, 0.3 * len(importances))))
    plt.barh(np.array(feature_names)[sorted_index], importances[sorted_index])
    plt.xlabel("Feature Importance Score")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()

    print("Evaluation complete. All metrics and plots generated.")
    return df

if __name__ == "__main__":
    evaluate_models()