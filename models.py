"""
Task 3 — Machine Learning Models Implementation
Author: Anthony Liao

• Loads preprocessed data from Task 2
• Trains three ML models:
    - Logistic Regression
    - Random Forest
    - XGBoost
• Measures training time and prediction time (validation + test)
• Returns trained models, timing results, and datasets for Task 4 evaluation
"""

import time
from preprocess import load_and_preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def time_model_prediction(model, X):
    """Returns predictions + how long they took."""
    start = time.time()
    preds = model.predict(X)
    end = time.time()
    return preds, end - start


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression + measure training time."""
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs"
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    
    return model, end - start


def train_random_forest(X_train, y_train):
    """Train a much faster Random Forest with competitive performance."""
    model = RandomForestClassifier(
        n_estimators=100,      # cut trees in half (200 → 100)
        max_depth=12,         # limit tree depth (massive speedup)
        min_samples_split=5,  # prevents overly deep trees
        min_samples_leaf=3,
        n_jobs=-1,            # use ALL CPU cores for speed
        class_weight="balanced",
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    return model, end - start


def train_xgboost(X_train, y_train):
    """Train XGBoost + measure training time."""
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    return model, end - start


def train_all_models(csv_path="creditcard_2023.csv"):
    """Train 3 ML models + measure training and prediction time."""

    print("Loading dataset and applying preprocessing...")
    data = load_and_preprocess_data(csv_path)
    print("Preprocessing complete.\n")

    X_train = data["X_train_sm"]
    y_train = data["y_train_sm"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]

    # ----- TRAINING MODELS -----
    print("Training Logistic Regression...")
    log_reg, logreg_train_time = train_logistic_regression(X_train, y_train)
    print(f"Logistic Regression trained. (Time: {logreg_train_time:.2f} sec)\n")

    print("Training Random Forest...")
    rf, rf_train_time = train_random_forest(X_train, y_train)
    print(f"Random Forest trained. (Time: {rf_train_time:.2f} sec)\n")

    print("Training XGBoost (this may take ~30–90 seconds)...")
    xgb, xgb_train_time = train_xgboost(X_train, y_train)
    print(f"XGBoost trained. (Time: {xgb_train_time:.2f} sec)\n")

    # ----- PREDICTION TIMES -----
    print("Measuring prediction times...")
    logreg_val_preds, logreg_val_time = time_model_prediction(log_reg, X_val)
    logreg_test_preds, logreg_test_time = time_model_prediction(log_reg, X_test)

    rf_val_preds, rf_val_time = time_model_prediction(rf, X_val)
    rf_test_preds, rf_test_time = time_model_prediction(rf, X_test)

    xgb_val_preds, xgb_val_time = time_model_prediction(xgb, X_val)
    xgb_test_preds, xgb_test_time = time_model_prediction(xgb, X_test)

    print("Prediction timing complete.\n")

    print("All models trained and timed successfully!")

    return {
        "log_reg": log_reg,
        "random_forest": rf,
        "xgboost": xgb,

        "timing": {
            "logistic_regression": {
                "train_time": logreg_train_time,
                "val_pred_time": logreg_val_time,
                "test_pred_time": logreg_test_time,
            },
            "random_forest": {
                "train_time": rf_train_time,
                "val_pred_time": rf_val_time,
                "test_pred_time": rf_test_time,
            },
            "xgboost": {
                "train_time": xgb_train_time,
                "val_pred_time": xgb_val_time,
                "test_pred_time": xgb_test_time,
            },
        },

        "data": data,
    }


if __name__ == "__main__":
    results = train_all_models()
    print("\nTiming summary:")
    print(results["timing"])
