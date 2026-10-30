"""End-to-end training entry point for the DMBI mini project."""

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from src.data_preprocessing import ROOT_DIR, load_data, describe_dataset
from src.evaluate_models import evaluate_models
from src.exploratory_analysis import generate_exploratory_figures
from src.predict_student import MODEL_PATH
from src.train_models import train_models

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/project.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> None:
    data = load_data()
    logging.info("Loaded dataset with shape %s", data.shape)
    summary = describe_dataset(data)
    print(f"Dataset dimensions: {summary['dimensions']}")
    print("\nFirst five records:\n", summary["head"])
    print("\nMissing values:\n", summary["missing_values"])
    print(f"\nDuplicate records: {summary['duplicate_records']}")
    print("\nTarget distribution (0=At Risk, 1=Pass):\n", summary["target_distribution"])
    generate_exploratory_figures(data)
    models, X_test, _, y_test, _ = train_models(data)
    threshold_frame = pd.read_csv("reports/threshold_analysis.csv")
    thresholds = threshold_frame.sort_values(
        ["At Risk Recall", "At Risk F1", "At Risk Precision"],
        ascending=False,
    ).drop_duplicates("Model").set_index("Model")["Threshold"].to_dict()
    results = evaluate_models(models, X_test, y_test, thresholds=thresholds)
    metadata = {
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_state": 42,
        "dataset_shape": list(data.shape),
        "feature_columns": list(X_test.columns),
        "dataset_sha256": hashlib.sha256(
            (ROOT_DIR / "data" / "raw" / "student-mat.csv").read_bytes()
        ).hexdigest(),
        "package_versions": {
            package: version(package)
            for package in ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn", "joblib"]
        },
        "final_test_metrics": results.to_dict(orient="records"),
        "thresholds": thresholds,
    }
    (ROOT_DIR / "models" / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print("\nFinal held-out test results:")
    print(results.to_string(index=False))
    print(f"\nSaved selected model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
