"""Exploratory analysis and chart generation using the real dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_preprocessing import ROOT_DIR, add_target, load_data

FIGURES_DIR = ROOT_DIR / "reports" / "figures"
sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_exploratory_figures(data: pd.DataFrame) -> None:
    """Generate the required exploratory figures from actual observations."""
    frame = add_target(data)
    labels = frame["target"].map({0: "At Risk", 1: "Pass"})

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x=labels, order=["At Risk", "Pass"], ax=ax, hue=labels, legend=False)
    ax.set(title="Student Result Distribution", xlabel="Result", ylabel="Number of students")
    _save(fig, "01_result_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=frame, x="studytime", y="G3", ax=ax)
    ax.set(title="Study Time vs Final Grade", xlabel="Weekly study time category", ylabel="Final grade (G3)")
    _save(fig, "02_study_time_vs_grade.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=frame, x="absences", y="G3", hue=labels, palette="Set1", ax=ax)
    ax.set(title="Absences vs Final Grade", xlabel="Number of absences", ylabel="Final grade (G3)")
    ax.legend(title="Result")
    _save(fig, "03_absences_vs_grade.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=frame, x="failures", y="G3", ax=ax)
    ax.set(title="Previous Failures vs Final Grade", xlabel="Number of previous failures", ylabel="Final grade (G3)")
    _save(fig, "04_failures_vs_grade.png")

    numeric = frame.select_dtypes(include="number").drop(columns=["target"], errors="ignore")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Variables")
    _save(fig, "05_correlation_heatmap.png")


if __name__ == "__main__":
    generate_exploratory_figures(load_data())
