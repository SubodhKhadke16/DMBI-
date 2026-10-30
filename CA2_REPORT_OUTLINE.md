# CA2 Mini Project Implementation Report

## Student Academic Performance Prediction System

**Subject:** Data Mining and Business Intelligence (DMBI)  
**Student name:** `[Enter your name]`  
**Roll number:** `[Enter roll number]`  
**Class/Division:** `[Enter class and division]`  
**Faculty:** `[Enter faculty name]`  
**Submission date:** `[Enter submission date]`

> **Submission note:** Export this document to PDF or DOCX after replacing the bracketed identity fields and inserting your own screenshots. Do not insert invented screenshots or change the actual output values.

## 1. Aim

To implement a Data Mining classification system that predicts whether a Mathematics student is likely to Pass or be At Risk, using the same UCI Student Performance case study analysed in CA1.

## 2. Introduction

CA2 converts the CA1 analysis into an executable Python solution. The implementation loads the official UCI dataset, creates a binary target from final grade `G3`, preprocesses numerical and categorical attributes, trains three classifiers, evaluates their predictions, saves the selected model, and provides command-line and Streamlit prediction interfaces.

## 3. Problem Statement and Real-World Goal

The system identifies students who may need additional academic support:

```text
Pass: G3 >= 10
At Risk: G3 < 10
```

The real-world goal is early advisory support. A teacher or institution can use the output to decide which students may benefit from a conversation, mentoring, tutoring, attendance follow-up, or other support.

## 4. Software and Hardware Requirements

### Software

- Windows 11
- Python 3.12.10 used for the verified run
- Visual Studio Code
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Pytest

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Hardware

The project is suitable for an ordinary Windows laptop or desktop with:

- At least 4 GB RAM recommended
- A modern dual-core or better processor
- Storage for Python packages and generated figures
- Internet access for the initial dataset download

## 5. Dataset Description

The official UCI Student Performance Mathematics dataset is stored at:

```text
data/raw/student-mat.csv
```

Verified dataset facts:

| Item | Value |
|---|---:|
| Rows | 395 |
| Columns | 33 |
| Missing values | 0 |
| Duplicate rows | 0 |
| At Risk | 130 |
| Pass | 265 |
| Separator | `;` |

The data contains school, demographic, family, academic behaviour, support, social, health, and attendance attributes.

## 6. Project Structure

```text
DMBI CA/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── data/raw/student-mat.csv
├── models/best_model.joblib
├── models/model_metadata.json
├── notebooks/student_performance_analysis.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── cross_validation_results.csv
│   ├── threshold_analysis.csv
│   ├── test_predictions.csv
│   └── figures/
├── logs/project.log
├── tests/
└── src/
```

## 7. Data Preprocessing

The CSV is loaded with Pandas and a semicolon separator. The target is created as:

```python
(data["G3"] >= 10).astype(int)
```

The following columns are removed from the model inputs:

```text
G1, G2, G3
```

This prevents the model from directly using earlier or final examination grades.

Numerical columns use:

- Median imputation
- StandardScaler

Categorical columns use:

- Most-frequent imputation
- OneHotEncoder

All preprocessing is contained in the Scikit-learn pipeline, so it is fitted inside training and cross-validation rather than on the complete dataset.

## 8. Algorithm Implementation

The implementation compares:

1. Decision Tree Classifier
2. Random Forest Classifier
3. Logistic Regression

The algorithms were selected because they provide an interpretable tree, a non-linear ensemble, and a simpler statistical baseline.

## 9. Model Training and Selection

The data uses an 80:20 stratified train/test split with `random_state=42`.

The training partition uses five-fold StratifiedKFold cross-validation. Controlled GridSearchCV searches model parameters using At Risk recall as the principal scoring measure. Probability thresholds 0.30, 0.40, 0.50, 0.60, and 0.70 are evaluated using training-only cross-validation predictions.

The test set is not used for hyperparameter tuning, threshold selection, or algorithm selection. It is used for the final evaluation.

## 10. Cross-Validation Results

Source: `reports/cross_validation_results.csv`.

| Model | Mean Accuracy | Mean At Risk Precision | Mean At Risk Recall | Mean At Risk F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 0.611 | 0.426 | 0.538 | 0.473 |
| Random Forest | 0.684 | 0.518 | 0.461 | 0.486 |
| Logistic Regression | 0.687 | 0.515 | **0.606** | **0.554** |

The selected threshold for each model in this executed run was 0.70. The Random Forest had the highest thresholded training cross-validation At Risk recall and was saved as the selected model.

## 11. Final Test Results

Source: `reports/model_comparison.csv`. These are final held-out test results, not expected values.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | At Risk Precision | At Risk Recall | At Risk F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.582 | 0.672 | 0.736 | 0.703 | 0.579 | 0.333 | 0.269 | 0.298 |
| Random Forest | **0.684** | 0.679 | **1.000** | **0.809** | **0.632** | **1.000** | 0.038 | 0.074 |
| Logistic Regression | 0.671 | **0.721** | 0.830 | 0.772 | 0.594 | 0.500 | **0.346** | **0.409** |

The Random Forest was selected using training-only cross-validation, not by inspecting the test results. Its low final test At Risk recall shows that model selection is uncertain on this small dataset. This must be stated honestly in the presentation.

## 12. Confusion Matrix Interpretation

The final test confusion matrices are saved in `reports/figures/`.

For the selected Random Forest at the selected threshold, the matrix was:

```text
Actual At Risk predicted At Risk: 1
Actual At Risk predicted Pass:    25
Actual Pass predicted At Risk:    0
Actual Pass predicted Pass:       53
```

This demonstrates why a single accuracy value is not enough. Although the model correctly classified many Pass students, it missed 25 of the 26 At Risk students in this test set. The model should therefore not be deployed without further validation and threshold/business review.

## 13. Visualizations

The project generates:

- Student result distribution
- Study time versus final grade
- Absences versus final grade
- Previous failures versus final grade
- Numerical correlation heatmap
- Model comparison
- Selected confusion matrix
- Feature importance
- ROC curve comparison
- Precision-Recall comparison
- Threshold analysis
- Separate model confusion matrices

Insert screenshots from `reports/figures/` in this section.

## 14. Prediction System

The command-line system is run with:

```powershell
python -m src.predict_student
```

The user enters student attributes and receives:

- Pass or At Risk
- Probability of Pass
- Probability of At Risk

The Streamlit dashboard is run with:

```powershell
streamlit run app.py
```

Insert the following actual screenshots:

- `[Insert screenshot of the dataset overview page]`
- `[Insert screenshot of the model comparison page]`
- `[Insert screenshot of the student prediction form]`
- `[Insert screenshot of the prediction result and probabilities]`

## 15. Testing and Verification

Automated tests are stored in `tests/` and run with:

```powershell
python -m pytest -q
```

Verified test result:

```text
4 passed in 66.33 seconds
```

The tests cover:

- Dataset dimensions and target creation
- Exclusion of `G1`, `G2`, and `G3`
- Model training
- Valid class predictions
- Probability bounds
- Probability totals

The dashboard was also smoke-tested locally and returned HTTP 200.

## 16. Business Applications

The implementation can support:

- Early academic counselling
- Tutor or mentor allocation
- Attendance follow-up
- Identification of students for additional study resources
- Aggregated planning of student-support programmes

The prediction is advisory. Teachers must review the student’s context and speak with the student before taking action.

## 17. Limitations and Ethical Use

- The dataset is small and historical.
- It may not represent the user’s institution.
- Predictions are statistical and not causal.
- The model may perform differently for demographic subgroups.
- A false negative may result in a student not receiving timely support.
- A false positive may cause unnecessary concern.
- Predictions must not be used for punishment, exclusion, grading, or disciplinary action.

## 18. Conclusion

The project implements the CA1 case study as a working Data Mining system. It uses a real dataset, leakage-safe preprocessing, three classification algorithms, training-only cross-validation, threshold analysis, final evaluation, visualizations, saved model artefacts, a CLI predictor, and a Streamlit dashboard. The implementation achieves the CA2 goal of demonstrating how Data Mining can support educational decision-making while also showing that the current small dataset requires careful interpretation and further validation.

## 19. References

1. UCI Machine Learning Repository, “Student Performance Dataset,” https://archive.ics.uci.edu/dataset/320/student+performance
2. Scikit-learn documentation, https://scikit-learn.org/stable/
3. Pandas documentation, https://pandas.pydata.org/docs/
4. Streamlit documentation, https://docs.streamlit.io/

## 20. Verified Data Dictionary and Feature Groups

| Feature group | Examples | Explanation |
|---|---|---|
| School and demographics | `school`, `sex`, `age`, `address` | Basic student and school context |
| Family background | `famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `guardian` | Family structure, education, employment, and guardian information |
| School access and behaviour | `traveltime`, `studytime`, `failures`, `absences` | Attendance, travel, study habits, and previous academic difficulty |
| Support and aspirations | `schoolsup`, `famsup`, `higher`, `internet` | Available educational support, higher-education intention, and access |
| Social and health context | `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `romantic` | Self-reported relationships, activities, alcohol ratings, health, and relationship status |
| Excluded grades | `G1`, `G2`, `G3` | Removed from model inputs to prevent grade leakage |

## 21. Verified Dataset Percentages

The 395 records contain:

| Target class | Count | Percentage |
|---|---:|---:|
| At Risk | 130 | 32.91% |
| Pass | 265 | 67.09% |

The class imbalance is why the implementation reports At Risk-specific precision, recall, F1-score, specificity, false-negative rate, and balanced accuracy rather than relying on accuracy alone.

## 22. Confusion-Matrix Explanation for Presentation

For the selected Random Forest configuration on the held-out test data:

- 1 At Risk student was correctly identified.
- 25 At Risk students were predicted as Pass.
- 0 Pass students were predicted as At Risk.
- 53 Pass students were correctly identified.

This result gives high At Risk precision but very low At Risk recall. In practical terms, the model was conservative about issuing an At Risk signal and missed many students who actually belonged to that class. This is why the result must be presented as an experiment requiring improvement, not as a deployment-ready intervention tool.

## 23. Model Interpretation Summary

The final test results show:

- Random Forest had the highest accuracy (0.684), overall F1-score (0.809), and ROC-AUC (0.632).
- Logistic Regression had the highest At Risk recall among the final thresholded test predictions (0.346) and the highest At Risk F1-score (0.409).
- Decision Tree had At Risk recall of 0.269 in the final test evaluation.
- The training cross-validation ranking was different from the final test ranking.

This difference demonstrates the importance of separating model selection from final evaluation and reporting uncertainty honestly.

## 24. Five-Minute Presentation Structure

1. **Problem:** Institutions need early indicators of students who may need support.
2. **Dataset:** Official UCI Mathematics student dataset, 395 records and 33 columns.
3. **Target:** Pass if `G3 >= 10`, otherwise At Risk.
4. **Leakage prevention:** Excluded `G1`, `G2`, and `G3`.
5. **Preprocessing:** Imputation, scaling, one-hot encoding, and pipeline-based fitting.
6. **Algorithms:** Decision Tree, Random Forest, and Logistic Regression.
7. **Evaluation:** Accuracy plus At Risk precision, recall, F1-score, confusion matrices, and ROC-AUC.
8. **Implementation:** Python pipeline, saved Joblib model, CLI predictor, and Streamlit dashboard.
9. **Business value:** Prioritises teacher review and support planning.
10. **Responsible use:** Advisory only; never use the prediction to punish or exclude a student.

## 25. Likely Teacher Questions and Answers

**Q: Why did you use three algorithms when only one was required?**  
A: CA2 requires at least one algorithm, but comparing three models provides a stronger analysis and shows the trade-offs between interpretability, ensemble learning, and a linear baseline.

**Q: Why is accuracy not the only metric?**  
A: The classes are imbalanced. A model can appear accurate by predicting the majority Pass class, while still missing many At Risk students.

**Q: Does the model prove that absences or study time cause failure?**  
A: No. Feature relationships and model importance indicate predictive patterns, not causation.

**Q: Can the model automatically decide a student's future?**  
A: No. The prediction is advisory and must be reviewed by teachers with the student's wider context.

**Q: What is the main limitation?**  
A: The dataset is small and historical, and the final test results differ from cross-validation results. External validation on a newer local cohort is needed.
