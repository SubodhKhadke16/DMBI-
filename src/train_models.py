"""Leakage-safe model selection, tuning, and final training."""

import json
from datetime import datetime, timezone
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    cross_validate,
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score

from src.data_preprocessing import ROOT_DIR, build_preprocessor, load_data, prepare_features

MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
CV_PATH = ROOT_DIR / "reports" / "cross_validation_results.csv"
PARAMS_PATH = ROOT_DIR / "reports" / "best_hyperparameters.json"
THRESHOLD_PATH = ROOT_DIR / "reports" / "threshold_analysis.csv"
RANDOM_STATE = 42
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def create_models() -> Dict[str, object]:
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    }


def _grids() -> Dict[str, dict]:
    return {
        "Decision Tree": {
            "model__max_depth": [3, 5, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [150, 300],
            "model__max_depth": [None, 8],
            "model__min_samples_leaf": [1, 2],
            "model__class_weight": [None, "balanced"],
        },
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs"],
            "model__class_weight": [None, "balanced"],
        },
    }


def _risk_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def _risk_precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=0, zero_division=0)


def _risk_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)


def _select_threshold(y_true: pd.Series, risk_probability) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLDS:
        predicted = (risk_probability >= threshold).astype(int)
        rows.append(
            {
                "Threshold": threshold,
                "At Risk Precision": _risk_precision(y_true, predicted),
                "At Risk Recall": _risk_recall(y_true, predicted),
                "At Risk F1": _risk_f1(y_true, predicted),
            }
        )
    analysis = pd.DataFrame(rows)
    selected = analysis.sort_values(
        ["At Risk Recall", "At Risk F1", "At Risk Precision"],
        ascending=False,
    ).iloc[0]["Threshold"]
    return float(selected), analysis


def train_models(
    data: pd.DataFrame,
) -> Tuple[Dict[str, Pipeline], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Tune on training folds, select by training CV, then fit final models."""
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "risk_precision": make_scorer(_risk_precision),
        "risk_recall": make_scorer(_risk_recall),
        "risk_f1": make_scorer(_risk_f1),
    }
    trained: Dict[str, Pipeline] = {}
    cv_rows = []
    params = {}
    threshold_rows = []
    thresholds = {}

    for name, estimator in create_models().items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X_train)), ("model", estimator)]
        )
        search = GridSearchCV(
            pipeline,
            _grids()[name],
            scoring=make_scorer(_risk_recall),
            cv=cv,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        trained[name] = best
        params[name] = search.best_params_
        cv_scores = cross_validate(best, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_rows.append(
            {
                "Model": name,
                "Mean Accuracy": cv_scores["test_accuracy"].mean(),
                "Std Accuracy": cv_scores["test_accuracy"].std(),
                "Mean At Risk Precision": cv_scores["test_risk_precision"].mean(),
                "Std At Risk Precision": cv_scores["test_risk_precision"].std(),
                "Mean At Risk Recall": cv_scores["test_risk_recall"].mean(),
                "Std At Risk Recall": cv_scores["test_risk_recall"].std(),
                "Mean At Risk F1": cv_scores["test_risk_f1"].mean(),
                "Std At Risk F1": cv_scores["test_risk_f1"].std(),
                "Best CV Score": search.best_score_,
            }
        )
        risk_probability = cross_val_predict(
            best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
        )[:, 0]
        selected_threshold, threshold_analysis = _select_threshold(y_train, risk_probability)
        thresholds[name] = selected_threshold
        selected_metrics = threshold_analysis.loc[
            threshold_analysis["Threshold"] == selected_threshold
        ].iloc[0]
        cv_rows[-1].update(
            {
                "Selected Threshold": selected_threshold,
                "CV At Risk Precision at Threshold": selected_metrics["At Risk Precision"],
                "CV At Risk Recall at Threshold": selected_metrics["At Risk Recall"],
                "CV At Risk F1 at Threshold": selected_metrics["At Risk F1"],
            }
        )
        threshold_analysis.insert(0, "Model", name)
        threshold_rows.append(threshold_analysis)

    cv_frame = pd.DataFrame(cv_rows)
    threshold_frame = pd.concat(threshold_rows, ignore_index=True)
    REPORTS_DIR = ROOT_DIR / "reports"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_frame.to_csv(CV_PATH, index=False)
    threshold_frame.to_csv(THRESHOLD_PATH, index=False)
    PARAMS_PATH.write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    selected_name = cv_frame.sort_values(
        ["CV At Risk Recall at Threshold", "CV At Risk F1 at Threshold"],
        ascending=False,
    ).iloc[0]["Model"]
    save_best_model(
        trained[selected_name],
        selected_name,
        threshold=thresholds[selected_name],
        feature_columns=list(X.columns),
    )
    return trained, X_test, X_train, y_test, y_train


def save_best_model(
    model: Pipeline,
    name: str,
    threshold: float = 0.5,
    feature_columns: list[str] | None = None,
) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": name,
            "threshold": threshold,
            "feature_columns": feature_columns or [],
        },
        MODEL_PATH,
    )


if __name__ == "__main__":
    from src.evaluate_models import evaluate_models

    dataset = load_data()
    trained_models, X_test, _, y_test, _ = train_models(dataset)
    results = evaluate_models(trained_models, X_test, y_test, save_outputs=True)
    print(results.to_string(index=False))
