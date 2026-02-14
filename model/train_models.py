import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# Directory where this file lives (i.e. the "model" folder)
MODEL_DIR = Path(__file__).resolve().parent
# Project root (parent of model/)
PROJECT_ROOT = MODEL_DIR.parent
# Default path to the heart disease CSV (inside project data/ folder)
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_dataset.csv"

# Class labels for heart disease target (0 = No disease, 1 = Disease)
CLASS_NAMES = ["No Heart Disease", "Heart Disease"]


def load_dataset(csv_path: Optional[Union[Path, str]] = None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the heart disease dataset from a CSV file.

    The CSV must have a "target" column (0 = no heart disease, 1 = heart disease).
    All other columns are used as features.
    """
    path = Path(csv_path) if csv_path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Please ensure data/heart_disease_dataset.csv exists in the project."
        )

    # Read raw CSV
    df = pd.read_csv(path)

    instances_in_dataset = df.shape[0]

    # Basic checks
    if "target" not in df.columns:
        raise ValueError('CSV must contain a "target" column.')

    # --- Simple preprocessing / cleaning ---
    # 1) Drop duplicate rows (if any)
    df = df.drop_duplicates().reset_index(drop=True)

    # 2) Fill missing numeric values with the column median
    #    (this is a simple and common way to handle missing values)
    df = df.fillna(df.median(numeric_only=True))

    # Split into features (X) and target (y)
    y = df["target"].copy()
    X = df.drop(columns=["target"])
    return X, y, instances_in_dataset


def split_dataset(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """Split the dataset into train and test parts."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def build_models():
    """
    Create all the classification models.

    Each model is wrapped in a simple Pipeline that:
    1) Scales all feature columns using StandardScaler
    2) Applies the actual classifier

    This helps many models (like Logistic Regression and kNN) train better
    and makes sure the same scaling is used later when we make predictions.
    """

    def make_scaled_model(estimator):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )

    models = {
        "Logistic Regression": make_scaled_model(LogisticRegression(max_iter=1000)),
        "Decision Tree": make_scaled_model(DecisionTreeClassifier(random_state=42)),
        "kNN": make_scaled_model(KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": make_scaled_model(GaussianNB()),
        "Random Forest (Ensemble)": make_scaled_model(
            RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        "XGBoost (Ensemble)": make_scaled_model(
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        ),
    }
    return models


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate all models.

    Returns:
        metrics_df: pandas DataFrame with all metrics.
        trained_models: dict of fitted model objects.
        reports: dict with classification reports (as text).
        confusion_matrices: dict with confusion matrices (numpy arrays).
    """
    rows = []
    trained_models = {}
    reports = {}
    confusion_matrices = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)

        # Some models expose predict_proba; for safety, check first
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            # Fallback: use decision_function if available, else set AUC to NaN
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                auc = roc_auc_score(y_test, scores)
            else:
                auc = np.nan

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        rows.append(
            {
                "ML Model Name": name,
                "Accuracy": acc,
                "AUC": auc,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "MCC": mcc,
            }
        )

        reports[name] = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    metrics_df = pd.DataFrame(rows)
    return metrics_df, trained_models, reports, confusion_matrices


def save_trained_models(trained_models, save_dir: Path = MODEL_DIR):
    """
    Save each trained model as a separate .pkl file in the model directory.

    Example file names:
        logistic_regression.pkl
        decision_tree.pkl
        knn.pkl
        naive_bayes.pkl
        random_forest_ensemble.pkl
        xgboost_ensemble.pkl
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, model in trained_models.items():
        # Make a simple, file-system-friendly name
        file_safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        file_path = save_dir / f"{file_safe_name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    # Simple command-line run to train and print metrics
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    models = build_models()
    metrics_df, trained_models, reports, confusion_matrices = evaluate_models(
        models, X_train, X_test, y_train, y_test
    )

    print("=== Dataset Shape ===")
    print(f"Total instances: {X.shape[0]}")
    print(f"Total features: {X.shape[1]}")
    print()

    print("=== Model Comparison Table ===")
    print(metrics_df.to_string(index=False))
    print()

    print("=== Classification Reports ===")
    for name, rep in reports.items():
        print(f"\n--- {name} ---")
        print(rep)

    # Save trained models as .pkl files inside the model directory
    save_trained_models(trained_models, MODEL_DIR)

    print("\n=== Saved Model Files (.pkl) in 'model' folder ===")
    for pkl_file in sorted(MODEL_DIR.glob("*.pkl")):
        print(f"- {pkl_file.name}")

