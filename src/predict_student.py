"""Interactive command-line prediction using the saved complete pipeline."""

from pathlib import Path
import joblib
import pandas as pd

from src.data_preprocessing import ROOT_DIR, load_data, prepare_features

MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"


def load_saved_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run `python main.py` first.")
    return joblib.load(MODEL_PATH)


def predict_student(values: dict) -> tuple[str, float, float]:
    bundle = load_saved_model()
    model = bundle["model"]
    prediction = int(model.predict(pd.DataFrame([values]))[0])
    probabilities = model.predict_proba(pd.DataFrame([values]))[0]
    return ("Pass" if prediction == 1 else "At Risk", float(probabilities[1]), float(probabilities[0]))


def run_cli() -> None:
    feature_frame, _ = prepare_features(load_data())
    reference = feature_frame.iloc[0]
    values = {}
    print("Enter values for a student. Press Enter to use the first dataset row as an example value.")
    for column in reference.index:
        raw = input(f"{column} [{reference[column]}]: ").strip()
        value = raw if raw else reference[column]
        if pd.api.types.is_numeric_dtype(feature_frame[column]):
            try:
                value = float(value) if "." in str(value) else int(value)
            except ValueError as exc:
                raise ValueError(f"{column} must be numeric.") from exc
        values[column] = value
    result, pass_probability, risk_probability = predict_student(values)
    print(f"\nPrediction: {result}")
    print(f"Probability of Pass: {pass_probability:.2%}")
    print(f"Probability of At Risk: {risk_probability:.2%}")
    print("Advisory only: teachers should review this result alongside student context.")


if __name__ == "__main__":
    run_cli()
