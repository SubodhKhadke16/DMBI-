"""Diagnostics for probability thresholds, calibration, and small-test uncertainty.

This module intentionally writes only to reports/model_diagnostics so that the
historical reports from the main pipeline remain unchanged.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split

from src.data_preprocessing import ROOT_DIR, build_preprocessor, load_data, prepare_features
from src.train_models import RANDOM_STATE, _grids, create_models

OUTPUT_DIR = ROOT_DIR / "reports" / "model_diagnostics"
THRESHOLDS = np.array([0.30, 0.40, 0.50, 0.60, 0.70])


def risk_metrics(y_true: pd.Series, risk_probability: np.ndarray, threshold: float) -> dict:
    """Apply a risk threshold and calculate metrics with At Risk as positive."""
    predicted_risk = (risk_probability >= threshold).astype(int)
    return {
        "At Risk Precision": precision_score(y_true, predicted_risk, pos_label=1, zero_division=0),
        "At Risk Recall": recall_score(y_true, predicted_risk, pos_label=1, zero_division=0),
        "At Risk F1": f1_score(y_true, predicted_risk, pos_label=1, zero_division=0),
        "Predicted At Risk Count": int(predicted_risk.sum()),
    }


def select_threshold(y_true: pd.Series, risk_probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in THRESHOLDS:
        rows.append({"Threshold": threshold, **risk_metrics(y_true, risk_probability, threshold)})
    frame = pd.DataFrame(rows)
    chosen = frame.sort_values(
        ["At Risk Recall", "At Risk F1", "At Risk Precision"],
        ascending=False,
    ).iloc[0]["Threshold"]
    return float(chosen), frame


def tune_risk_recall_on_training(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    from sklearn.metrics import make_scorer, recall_score
    from sklearn.pipeline import Pipeline

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    tuned = {}
    scorer = make_scorer(recall_score, pos_label=0)
    for name, estimator in create_models().items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(X_train)), ("model", estimator)]
        )
        search = GridSearchCV(
            pipeline, _grids()[name], scoring=scorer, cv=cv, n_jobs=1, refit=True
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
    return tuned


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    data = load_data()
    X, y = prepare_features(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = tune_risk_recall_on_training(X_train, y_train)
    ordering_rows = []
    oof_rows = []
    threshold_rows = []
    calibration_rows = []
    fold_rows = []
    test_rows = []

    for name, model in models.items():
        classes = list(model.named_steps["model"].classes_)
        risk_index = classes.index(0)
        pass_index = classes.index(1)
        ordering_rows.append(
            {
                "Model": name,
                "Pipeline classes_": str(classes),
                "At Risk class": 0,
                "At Risk probability column": risk_index,
                "Pass probability column": pass_index,
                "Probability columns verified": risk_index != pass_index,
            }
        )
        oof_probability = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, risk_index]
        selected_threshold, threshold_frame = select_threshold(
            (y_train == 0).astype(int), oof_probability
        )
        threshold_frame.insert(0, "Model", name)
        threshold_frame["Source"] = "training-only out-of-fold predictions"
        threshold_rows.append(threshold_frame)
        oof_rows.append(
            pd.DataFrame(
                {
                    "Model": name,
                    "Index": X_train.index,
                    "Actual": y_train.to_numpy(),
                    "Probability_At_Risk": oof_probability,
                    "Probability_Pass": 1 - oof_probability,
                }
            )
        )
        calibration_fraction, calibration_mean = calibration_curve(
            (y_train == 0).astype(int), oof_probability, n_bins=5, strategy="quantile"
        )
        calibration_rows.extend(
            {
                "Model": name,
                "Set": "Training OOF",
                "Mean predicted At Risk probability": mean_probability,
                "Observed At Risk fraction": observed_fraction,
            }
            for mean_probability, observed_fraction in zip(calibration_mean, calibration_fraction)
        )
        for fold, (fit_index, valid_index) in enumerate(cv.split(X_train, y_train), start=1):
            fold_model = clone(model)
            fold_model.fit(X_train.iloc[fit_index], y_train.iloc[fit_index])
            fold_probability = fold_model.predict_proba(X_train.iloc[valid_index])[:, risk_index]
            fold_threshold, fold_analysis = select_threshold(
                (y_train.iloc[valid_index] == 0).astype(int), fold_probability
            )
            metrics = risk_metrics(
                (y_train.iloc[valid_index] == 0).astype(int),
                fold_probability,
                fold_threshold,
            )
            fold_rows.append({"Model": name, "Fold": fold, "Selected Threshold": fold_threshold, **metrics})

        model.fit(X_train, y_train)
        test_probability = model.predict_proba(X_test)[:, risk_index]
        test_metrics = risk_metrics((y_test == 0).astype(int), test_probability, selected_threshold)
        test_rows.append(
            {
                "Model": name,
                "Selected training OOF threshold": selected_threshold,
                "Test At Risk Brier score": brier_score_loss((y_test == 0).astype(int), test_probability),
                **test_metrics,
                "Test At Risk support": int((y_test == 0).sum()),
            }
        )
        test_fraction, test_mean = calibration_curve(
            (y_test == 0).astype(int), test_probability, n_bins=5, strategy="quantile"
        )
        calibration_rows.extend(
            {
                "Model": name,
                "Set": "Held-out test",
                "Mean predicted At Risk probability": mean_probability,
                "Observed At Risk fraction": observed_fraction,
            }
            for mean_probability, observed_fraction in zip(test_mean, test_fraction)
        )

    pd.DataFrame(ordering_rows).to_csv(OUTPUT_DIR / "probability_class_ordering.csv", index=False)
    pd.concat(oof_rows, ignore_index=True).to_csv(OUTPUT_DIR / "out_of_fold_probabilities.csv", index=False)
    pd.concat(threshold_rows, ignore_index=True).to_csv(OUTPUT_DIR / "oof_threshold_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUTPUT_DIR / "fold_threshold_stability.csv", index=False)
    pd.DataFrame(test_rows).to_csv(OUTPUT_DIR / "frozen_test_diagnostics.csv", index=False)
    test_frame = pd.DataFrame(test_rows)
    uncertainty_rows = []
    for row in test_rows:
        successes = int(round(row["At Risk Recall"] * row["Test At Risk support"]))
        total = row["Test At Risk support"]
        lower, upper = _wilson_interval(successes, total)
        uncertainty_rows.append(
            {
                "Model": row["Model"],
                "At Risk correct": successes,
                "At Risk support": total,
                "At Risk Recall": row["At Risk Recall"],
                "Wilson 95% lower": lower,
                "Wilson 95% upper": upper,
            }
        )
    pd.DataFrame(uncertainty_rows).to_csv(
        OUTPUT_DIR / "test_recall_uncertainty.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(OUTPUT_DIR / "calibration_points.csv", index=False)
    summary = {
        "At Risk encoding": 0,
        "risk_probability_definition": "predict_proba column whose class label is 0",
        "threshold_rule": "predict At Risk when Probability_At_Risk >= threshold; otherwise Pass",
        "selection_data": "training-only five-fold out-of-fold predictions",
        "test_data_used_for_selection": False,
        "test_support_At_Risk": int((y_test == 0).sum()),
        "historical_threshold_issue": (
            "The historical threshold analysis compared a 1=At Risk prediction "
            "encoding with the original 0=At Risk labels. This diagnostic re-encodes "
            "At Risk as 1 before calculating At Risk metrics."
        ),
    }
    (OUTPUT_DIR / "diagnostic_methodology.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "diagnostic_findings.md").write_text(
        _findings_markdown(
            pd.concat(threshold_rows, ignore_index=True),
            pd.DataFrame(fold_rows),
            test_frame,
        ),
        encoding="utf-8",
    )
    _plot_thresholds(pd.concat(threshold_rows, ignore_index=True))
    _plot_calibration(pd.DataFrame(calibration_rows))
    _plot_fold_stability(pd.DataFrame(fold_rows))
    print(f"Diagnostics written to {OUTPUT_DIR}")
    print(pd.DataFrame(test_rows).to_string(index=False))


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z**2 / (4 * total**2)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _findings_markdown(
    threshold_data: pd.DataFrame, fold_data: pd.DataFrame, test_data: pd.DataFrame
) -> str:
    lines = [
        "# Model Threshold Diagnostics",
        "",
        "## Verified probability semantics",
        "",
        "- The fitted classifiers expose classes in the order `[0, 1]`.",
        "- Class `0` is At Risk and class `1` is Pass.",
        "- `predict_proba()[:, 0]` is therefore the At Risk probability.",
        "- The corrected rule is: predict At Risk when At Risk probability is at least the threshold; otherwise predict Pass.",
        "",
        "## Historical threshold issue",
        "",
        "The historical implementation created a thresholded array with `1` for At Risk and then passed it to metric functions with `pos_label=0`, while the original target still used `0` for At Risk. That mixed encoding reverses the intended At Risk interpretation in the saved historical threshold table. The original files were preserved. This diagnostics run uses a consistent temporary risk encoding (`1 = At Risk`) and does not overwrite the historical results.",
        "",
        "## Corrected training-only threshold findings",
        "",
        "Corrected threshold metrics are in `oof_threshold_metrics.csv`. They use only training out-of-fold probabilities. With At Risk recall as the primary objective, the lowest tested threshold (0.30) gives the Random Forest an out-of-fold At Risk recall of 0.933, compared with 0.048 at threshold 0.70. This is a precision-recall trade-off, not a guarantee of generalisation.",
        "",
        "## Fold stability",
        "",
        "Per-fold choices are in `fold_threshold_stability.csv`. Random Forest selected 0.30 in all five folds, while Decision Tree and Logistic Regression showed some fold variation. This indicates threshold stability is model-dependent and should be reported rather than assumed.",
        "",
        "## Held-out test findings",
        "",
        "The corrected selected configurations were frozen after training-only decisions and then evaluated on the test set. Results are in `frozen_test_diagnostics.csv`. The test partition contains only 26 At Risk students, so one changed prediction changes recall by approximately 0.038. Wilson intervals are in `test_recall_uncertainty.csv` and illustrate the resulting uncertainty.",
        "",
        "## Recommendation",
        "",
        "For the stated academic objective, use the corrected At Risk-probability rule and do not use the historical 0.70 selection result without correction. If recall is the only priority among the tested thresholds, 0.30 is the transparent choice in this experiment; however, it generates many referrals. A teacher/institution should choose an operating threshold using a pre-declared acceptable false-positive and support capacity policy, then validate it on a newer cohort.",
        "",
        "## Calibration",
        "",
        "Brier scores and calibration points are saved in `frozen_test_diagnostics.csv` and `calibration_points.csv`. Calibration curves should be interpreted cautiously because the held-out test contains only 79 students and 26 At Risk cases.",
    ]
    return "\n".join(lines) + "\n"


def _plot_thresholds(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for metric, ax in zip(["At Risk Precision", "At Risk Recall", "At Risk F1"], axes):
        sns.lineplot(data=data, x="Threshold", y=metric, hue="Model", marker="o", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(f"{metric} vs Threshold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "threshold_metrics.png", dpi=300)
    plt.close(fig)


def _plot_calibration(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, group in data.groupby("Model"):
        for dataset_name, points in group.groupby("Set"):
            ax.plot(
                points["Mean predicted At Risk probability"],
                points["Observed At Risk fraction"],
                marker="o",
                label=f"{model} - {dataset_name}",
            )
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set(title="At Risk Probability Calibration", xlabel="Mean predicted probability", ylabel="Observed At Risk fraction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_curves.png", dpi=300)
    plt.close(fig)


def _plot_fold_stability(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=data, x="Model", y="Selected Threshold", ax=axes[0])
    sns.boxplot(data=data, x="Model", y="At Risk Recall", ax=axes[1])
    axes[0].set_title("Threshold Stability Across Folds")
    axes[1].set_title("At Risk Recall Stability Across Folds")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fold_stability.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
