"""Data loading and leakage-safe preprocessing for the UCI student dataset."""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "student-mat.csv"
EXCLUDED_COLUMNS = ["G1", "G2", "G3"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the official semicolon-separated UCI Mathematics CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download student-mat.csv from "
            "https://archive.ics.uci.edu/dataset/320/student+performance "
            "and place it in data/raw."
        )
    return pd.read_csv(path, sep=";")


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """Create the business target: Pass when G3 is at least 10."""
    if "G3" not in data.columns:
        raise ValueError("The dataset must contain the G3 final-grade column.")
    result = data.copy()
    result["target"] = (result["G3"] >= 10).astype(int)
    return result


def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Remove grade leakage and separate predictors from the binary target."""
    prepared = add_target(data)
    X = prepared.drop(columns=EXCLUDED_COLUMNS + ["target"])
    y = prepared["target"].rename("target")
    return X, y


def get_feature_columns(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Return numerical and categorical columns using pandas dtypes."""
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build transformers that are fitted only inside the training pipeline."""
    numeric, categorical = get_feature_columns(X)
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def describe_dataset(data: pd.DataFrame) -> dict:
    """Return observable dataset checks used by reports and the dashboard."""
    return {
        "dimensions": data.shape,
        "head": data.head(),
        "missing_values": data.isna().sum(),
        "duplicate_records": int(data.duplicated().sum()),
        "dtypes": data.dtypes.astype(str),
        "target_distribution": add_target(data)["target"].value_counts().sort_index(),
    }

