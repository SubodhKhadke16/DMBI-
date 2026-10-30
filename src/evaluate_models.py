"""Evaluate trained pipelines and create model/result visualizations."""

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    roc_curve,
    precision_recall_curve,
)

from src.data_preprocessing import ROOT_DIR

REPORTS_DIR = ROOT_DIR / "reports"


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_outputs: bool = True,
    thresholds: Dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    predictions = []
    reports = {}
    for name, model in models.items():
        probability_pass = model.predict_proba(X_test)[:, 1]
        probability_risk = 1 - probability_pass
        threshold = (thresholds or {}).get(name, 0.5)
        predicted = (probability_risk < threshold).astype(int)
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predicted),
                "Precision": precision_score(y_test, predicted, zero_division=0),
                "Recall": recall_score(y_test, predicted, zero_division=0),
                "F1-score": f1_score(y_test, predicted, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, probability_pass),
                "Average Precision": average_precision_score(y_test, probability_pass),
                "Balanced Accuracy": balanced_accuracy_score(y_test, predicted),
                "Specificity": tn / (tn + fp) if tn + fp else 0,
                "False Negative Rate": fn / (fn + tp) if fn + tp else 0,
                "False Positive Rate": fp / (fp + tn) if fp + tn else 0,
                "MCC": matthews_corrcoef(y_test, predicted),
                "At Risk Precision": precision_score(y_test, predicted, pos_label=0, zero_division=0),
                "At Risk Recall": recall_score(y_test, predicted, pos_label=0, zero_division=0),
                "At Risk F1": f1_score(y_test, predicted, pos_label=0, zero_division=0),
            }
        )
        report = classification_report(y_test, predicted, target_names=["At Risk", "Pass"], zero_division=0)
        reports[name] = classification_report(
            y_test, predicted, target_names=["At Risk", "Pass"], output_dict=True, zero_division=0
        )
        print(f"\n{name}\n{report}\nConfusion matrix:\n{confusion_matrix(y_test, predicted)}")
        if save_outputs:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = name.lower().replace(" ", "_")
            (REPORTS_DIR / f"classification_report_{safe_name}.txt").write_text(
                report, encoding="utf-8"
            )
            _save_confusion_matrix(y_test, predicted, name, f"confusion_matrix_{safe_name}.png")
        predictions.append(pd.DataFrame({"Actual": y_test, "Predicted": predicted, "Probability_Pass": probability_pass, "Probability_At_Risk": probability_risk}, index=X_test.index))

    results = pd.DataFrame(rows)
    if save_outputs:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
        (REPORTS_DIR / "classification_reports.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        best = results.sort_values(["At Risk Recall", "At Risk F1"], ascending=False).iloc[0]["Model"]
        predictions[list(models).index(best)].to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
        _save_comparison_chart(results)
        best_model = models[best]
        best_pred = best_model.predict(X_test)
        _save_confusion_matrix(y_test, best_pred, best, "07_selected_confusion_matrix.png")
        _save_feature_importance(best_model, best)
        _save_probability_curves(models, X_test, y_test)
        _save_threshold_chart()
    return results


def _save_comparison_chart(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_data = results.melt(id_vars="Model", value_vars=["Accuracy", "At Risk Precision", "At Risk Recall", "At Risk F1"])
    sns.barplot(data=plot_data, x="Model", y="value", hue="variable", ax=ax)
    ax.set(title="Algorithm Performance Comparison", xlabel="Algorithm", ylabel="Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric", loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "06_model_comparison.png", dpi=300)
    plt.close(fig)


def _save_confusion_matrix(
    y_test: pd.Series,
    predicted: pd.Series,
    model_name: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, predicted), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(title=f"Confusion Matrix - {model_name}", xlabel="Predicted label", ylabel="Actual label")
    ax.set_xticklabels(["At Risk", "Pass"])
    ax.set_yticklabels(["At Risk", "Pass"], rotation=0)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / filename, dpi=300)
    plt.close(fig)


def _save_feature_importance(model: object, model_name: str) -> None:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.Series(estimator.feature_importances_, index=names).nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    importance.plot.barh(ax=ax, color="#2a9d8f")
    ax.set(title=f"Top Feature Importances - {model_name}", xlabel="Importance", ylabel="Feature")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "08_feature_importance.png", dpi=300)
    plt.close(fig)


def _save_probability_curves(models: Dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        probability = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probability)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probability):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(title="ROC Curve Comparison", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "09_roc_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        probability = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, probability)
        ax.plot(recall, precision, label=name)
    ax.set(title="Precision-Recall Curve Comparison", xlabel="Recall (Pass)", ylabel="Precision (Pass)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "10_precision_recall_comparison.png", dpi=300)
    plt.close(fig)


def _save_threshold_chart() -> None:
    path = REPORTS_DIR / "threshold_analysis.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 6))
    for model, group in data.groupby("Model"):
        ax.plot(group["Threshold"], group["At Risk Recall"], marker="o", label=model)
    ax.set(title="At Risk Recall by Probability Threshold", xlabel="At Risk probability threshold", ylabel="At Risk recall")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "11_threshold_recall.png", dpi=300)
    plt.close(fig)
