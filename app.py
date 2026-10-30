"""Streamlit dashboard for the actual trained student-performance model."""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.data_preprocessing import ROOT_DIR, load_data, prepare_features

MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
RESULTS_PATH = ROOT_DIR / "reports" / "model_comparison.csv"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

st.set_page_config(page_title="Student Academic Risk Analytics", page_icon="🎓", layout="wide")
st.title("Student Academic Performance Prediction")
st.caption("An educational analytics dashboard using the UCI Student Performance dataset.")

try:
    data = load_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.header("Project overview")
st.sidebar.write("Predicts Pass (G3 ≥ 10) or At Risk (G3 < 10) without using earlier grades G1/G2.")
tab_overview, tab_models, tab_predict = st.tabs(["Overview", "Models & Visualizations", "Student Prediction"])

with tab_overview:
    st.subheader("Dataset statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", data.shape[0])
    c2.metric("Attributes", data.shape[1])
    c3.metric("Duplicate rows", int(data.duplicated().sum()))
    distribution = data.assign(Result=data["G3"].ge(10).map({True: "Pass", False: "At Risk"}))["Result"].value_counts()
    st.bar_chart(distribution)
    st.dataframe(data.head())

with tab_models:
    if not RESULTS_PATH.exists():
        st.warning("Run `python main.py` before opening model results.")
    else:
        results = pd.read_csv(RESULTS_PATH)
        st.subheader("Actual model comparison")
        st.dataframe(results.style.format({column: "{:.3f}" for column in results.columns if column != "Model"}))
    for image in sorted(FIGURES_DIR.glob("*.png")):
        st.image(str(image), caption=image.stem, use_container_width=True)

with tab_predict:
    if not MODEL_PATH.exists():
        st.warning("Run `python main.py` before making predictions.")
    else:
        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        st.write(f"Selected model: **{bundle['model_name']}**")
        X, _ = prepare_features(data)
        values = {}
        columns = st.columns(3)
        for index, column in enumerate(X.columns):
            default = X[column].median() if pd.api.types.is_numeric_dtype(X[column]) else str(X[column].mode().iloc[0])
            with columns[index % 3]:
                if pd.api.types.is_numeric_dtype(X[column]):
                    values[column] = st.number_input(column, value=float(default))
                else:
                    options = sorted(X[column].dropna().astype(str).unique().tolist())
                    values[column] = st.selectbox(column, options, index=options.index(str(default)) if str(default) in options else 0)
        if st.button("Predict student result", type="primary"):
            row = pd.DataFrame([values])
            result = int(model.predict(row)[0])
            probabilities = model.predict_proba(row)[0]
            st.success("Prediction: Pass" if result else "Prediction: At Risk")
            st.metric("Probability of Pass", f"{probabilities[1]:.2%}")
            st.metric("Probability of At Risk", f"{probabilities[0]:.2%}")
            st.info("Predictions are advisory and should be reviewed by teachers.")

