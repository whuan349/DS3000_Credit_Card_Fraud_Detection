'''
Load dataset

Stratified 80/10/10 split

Min–Max scale Amount only

SMOTE on train only

Returns everything for models & plots
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE


def load_dataset(csv_path: str = "creditcard_2023.csv"):
    """
    Load the credit card dataset and return:
    - features X (drop id + Class)
    - labels y (Class)
    """

    df = pd.read_csv(csv_path)

    # Drop an ID column if it exists (not useful for modelling)
    cols_to_drop = []
    for c in ["id", "ID", "Id"]:
        if c in df.columns:
            cols_to_drop.append(c)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Separate features and target
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column for labels.")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return df, X, y


def stratified_split(X, y, random_state: int = 42):
    """
    Perform an 80/10/10 **stratified** split:
    - 80% train
    - 10% validation
    - 10% test
    """

    # 80% train, 20% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )

    # Split temp into 10% val, 10% test (each is 50% of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_amount_feature(X_train, X_val, X_test):
    """
    Min–Max scale the 'Amount' column **only** (PCA components are already scaled).

    Returns:
    - X_train_scaled, X_val_scaled, X_test_scaled
    - fitted MinMaxScaler (in case we need it later)
    """

    if "Amount" not in X_train.columns:
        # If dataset does not have Amount for some reason, just return as-is
        return X_train, X_val, X_test, None

    scaler = MinMaxScaler()

    # Fit on train only
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled["Amount"] = scaler.fit_transform(
        X_train[["Amount"]]
    )
    X_val_scaled["Amount"] = scaler.transform(
        X_val[["Amount"]]
    )
    X_test_scaled["Amount"] = scaler.transform(
        X_test[["Amount"]]
    )

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE **only on the training set** to handle class imbalance.

    Returns:
    - X_train_sm, y_train_sm
    - fitted SMOTE object
    """

    sm = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

    return X_train_sm, y_train_sm, sm


def load_and_preprocess_data(csv_path: str = "creditcard_2023.csv"):
    """
    Convenience function that:
    1. Loads the dataset
    2. Performs stratified 80/10/10 split
    3. Scales Amount with Min–Max (train only)
    4. Applies SMOTE to the training set only

    Returns a dictionary with:
    - df_raw
    - X_train, X_val, X_test (scaled)
    - y_train, y_val, y_test
    - X_train_sm, y_train_sm (after SMOTE)
    - scaler, smote (fitted objects)
    """

    # 1. Load
    df_raw, X, y = load_dataset(csv_path)

    # 2. Stratified split
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    # 3. Scale Amount
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_amount_feature(
        X_train, X_val, X_test
    )

    # 4. SMOTE on training only
    X_train_sm, y_train_sm, smote = apply_smote(X_train_scaled, y_train)

    # Package everything
    return {
    "df_raw": df_raw,
    "X_train": X_train_scaled,
    "X_val": X_val_scaled,
    "X_test": X_test_scaled,
    "y_train": y_train,
    "y_val": y_val,
    "y_test": y_test,
    "X_train_sm": X_train_sm, 
    "y_train_sm": y_train_sm,   
    "scaler": scaler,
    "smote": smote,
}

