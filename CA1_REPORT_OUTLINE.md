# CA1 Case Study-Based Analysis Report

## From Data to Decisions: Student Academic Risk Prediction

**Subject:** Data Mining and Business Intelligence (DMBI)  
**Student name:** `[Enter your name]`  
**Roll number:** `[Enter roll number]`  
**Class/Division:** `[Enter class and division]`  
**Faculty:** `[Enter faculty name]`  
**Submission date:** 31 October 2026

> **Submission note:** Export this Markdown document to PDF after replacing the bracketed identity fields and adding the selected charts from `reports/figures/`. The numerical results in this report are from the executed project outputs and must not be changed without rerunning the pipeline.

## 1. Abstract

Educational institutions need to identify students who may require academic support before final results are available. This case study analyses the UCI Student Performance Mathematics dataset and frames the problem as binary classification. A student is labelled **Pass** when the final Mathematics grade (`G3`) is at least 10 and **At Risk** when `G3` is below 10. The early-support model excludes `G1`, `G2`, and `G3` from the predictors so that the final result is not directly revealed.

The dataset contains 395 student records and 33 columns. It has 130 At Risk students and 265 Pass students, with no missing values and no duplicate rows. Decision Tree, Random Forest, and Logistic Regression are suitable candidate algorithms because they provide an interpretable tree, an ensemble model, and a linear baseline. The project uses leakage-safe preprocessing, five-fold cross-validation on the training data, and a final untouched test set. At Risk recall is prioritised because failing to identify a student who needs support is more harmful than referring an additional student for review.

## 2. Introduction

Student performance is influenced by attendance, study habits, previous failures, family support, school support, demographics, and social factors. Analysing these attributes can help teachers and institutions plan early academic interventions. The purpose of this case study is not to replace teachers or make automatic decisions. It is to show how Data Mining can convert student information into an advisory risk signal.

## 3. Problem Statement

Build a classification solution that predicts whether a student is likely to Pass or be At Risk using non-final-grade academic, demographic, family, support, attendance, and behavioural attributes.

The business question is:

> Can student information available before the final examination be used to identify students who may need additional academic support?

## 4. Business Objectives

1. Identify students who may require early academic support.
2. Prioritise recall for the At Risk class.
3. Help teachers allocate mentoring, counselling, tutoring, or attendance follow-up.
4. Compare interpretable and ensemble Data Mining algorithms.
5. Present results in a form that supports human review rather than automatic punishment.

## 5. Data Source and Dataset Description

The source is the official [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance). This report uses `student-mat.csv`, the Mathematics student file.

| Property | Verified value |
|---|---:|
| Records | 395 |
| Original columns | 33 |
| Separator | Semicolon (`;`) |
| Missing values | 0 |
| Duplicate rows | 0 |
| At Risk students | 130 |
| Pass students | 265 |

Important groups of variables include:

- Demographic: school, sex, age, address
- Family: family size, parental education, occupation, guardian
- Academic behaviour: study time, failures, absences
- Support: school support, family support, internet access
- Lifestyle: free time, going out, alcohol consumption, health
- Grades: `G1`, `G2`, and `G3`

## 6. Data Understanding

The dataset was loaded with Pandas using the semicolon separator. The first five records, dimensions, data types, missing-value counts, duplicate count, and target distribution were inspected by `main.py`.

The target distribution is imbalanced: 130 of 395 students are At Risk and 265 are Pass. Therefore, accuracy alone is not sufficient. At Risk precision, At Risk recall, At Risk F1-score, confusion matrices, and false-negative rate are important.

## 7. Data Preprocessing

The target is created as:

```python
target = (G3 >= 10).astype(int)
```

where:

```text
0 = At Risk
1 = Pass
```

`G1`, `G2`, and `G3` are excluded from model inputs:

- `G1` and `G2` would reveal earlier examination performance.
- `G3` is the final grade used to create the target.

Numerical preprocessing:

1. Median imputation
2. StandardScaler

Categorical preprocessing:

1. Most-frequent imputation
2. OneHotEncoder with unknown-category handling

The transformations are inside a Scikit-learn `Pipeline` and `ColumnTransformer`. Thus, preprocessing is fitted within the training workflow and is not fitted on the complete dataset before the split.

## 8. Data Mining Task

This is a **supervised binary classification** task:

```text
Input: student demographic, behavioural, support and attendance attributes
Output: Pass or At Risk
```

## 9. Algorithms and Justification

### 9.1 Decision Tree Classifier

A Decision Tree splits data using rules and is easy to explain to non-technical stakeholders. It is useful when the institution wants understandable decision paths, but it can be sensitive to the training sample and may overfit.

### 9.2 Random Forest Classifier

Random Forest combines many decision trees. It can model non-linear relationships and is generally more robust than one tree. It is useful for comparing an ensemble approach, although individual predictions are less transparent.

### 9.3 Logistic Regression

Logistic Regression provides a strong, interpretable linear baseline for binary classification. It is useful for understanding whether a simpler model performs competitively.

At least two algorithms are required for CA1; this project evaluates three to make the comparison stronger.

## 10. Data Mining Process

The complete process is:

```text
Problem
  -> Data source
  -> Data understanding
  -> Target construction
  -> Leakage prevention
  -> Imputation, scaling and encoding
  -> Training-only cross-validation and tuning
  -> Threshold analysis
  -> Final test evaluation
  -> Insights
  -> Teacher-supported decision
```

The data is split into 80% training and 20% testing using stratification and `random_state=42`. The training partition is used for five-fold cross-validation and tuning. The test partition is kept untouched until final evaluation.

## 11. Expected and Available Results

### Expected results before execution

The expected output was a comparison of the three classifiers using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and At Risk-specific measures. No exact performance value was assumed before execution.

### Training-only cross-validation results

The actual values below come from [cross_validation_results.csv](reports/cross_validation_results.csv).

| Model | Mean Accuracy | Mean At Risk Precision | Mean At Risk Recall | Mean At Risk F1 |
|---|---:|---:|---:|---:|
| Decision Tree | 0.611 | 0.426 | 0.538 | 0.473 |
| Random Forest | 0.684 | 0.518 | 0.461 | 0.486 |
| Logistic Regression | 0.687 | 0.515 | **0.606** | **0.554** |

### Final held-out test results

The actual values below come from [model_comparison.csv](reports/model_comparison.csv).

| Model | Accuracy | ROC-AUC | At Risk Precision | At Risk Recall | At Risk F1 |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.582 | 0.579 | 0.333 | 0.269 | 0.298 |
| Random Forest | **0.684** | **0.632** | **1.000** | 0.038 | 0.074 |
| Logistic Regression | 0.671 | 0.594 | 0.500 | **0.346** | **0.409** |

The Random Forest was selected by the training-only thresholded cross-validation procedure. However, the held-out test result shows At Risk recall of only 0.038 for that selected configuration. This difference is an important finding: the dataset is small, and validation performance is not guaranteed to generalise to the held-out sample. The result should therefore be treated as an advisory experiment, not as evidence that the model is ready for high-stakes deployment.

## 12. Evaluation Measures

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Of the students predicted as a class, how many truly belong to that class.
- **Recall:** Of the students truly belonging to a class, how many were found.
- **F1-score:** Balance between precision and recall.
- **ROC-AUC:** Ranking quality across probability thresholds.
- **At Risk recall:** Most important operational measure because it measures how many students needing support were identified.
- **False-negative rate:** Proportion of actual At Risk students incorrectly predicted as Pass.
- **Specificity:** Ability to identify Pass students correctly.
- **Balanced accuracy:** Average of class-specific recalls.

## 13. Business Insights and Decision Support

1. The target distribution is imbalanced, so a model cannot be judged by accuracy alone.
2. The cross-validation results and final test results differ, showing that the small dataset creates uncertainty.
3. A predicted At Risk result should trigger teacher review, not an automatic decision.
4. Potential support actions include tutoring, attendance follow-up, mentoring, counselling, or a conversation with the student.
5. The model should support resource prioritisation while teachers consider context that is not present in the dataset.

## 14. Limitations and Ethical Considerations

- The dataset contains only 395 Mathematics records.
- It represents a particular historical educational context and may not generalise to another institution.
- Some fields are self-reported.
- The model detects statistical patterns and does not prove causes.
- A prediction may be wrong and must not be used to punish, label, or exclude a student.
- Performance should be reviewed across demographic subgroups before operational use.
- A newer local cohort should be used for external validation.

## 15. Conclusion

This case study demonstrates the complete Data Mining path from a real educational problem to a classification-based decision-support concept. UCI student data was inspected and prepared without using final or earlier grades as predictors. Three appropriate algorithms were compared, and At Risk recall was treated as the key business concern. The actual results show both the usefulness and the uncertainty of the approach. The responsible conclusion is that such a model can help prioritise teacher attention, but it cannot replace professional judgement or be used as an automatic student decision system.

## 16. References

1. UCI Machine Learning Repository, “Student Performance Dataset,” Dataset 320, https://archive.ics.uci.edu/dataset/320/student+performance
2. Scikit-learn documentation, https://scikit-learn.org/stable/
3. Pandas documentation, https://pandas.pydata.org/docs/
4. Matplotlib documentation, https://matplotlib.org/stable/
5. Seaborn documentation, https://seaborn.pydata.org/

## Appendix: Suggested figures to insert

- `reports/figures/01_result_distribution.png`
- `reports/figures/02_study_time_vs_grade.png`
- `reports/figures/03_absences_vs_grade.png`
- `reports/figures/05_correlation_heatmap.png`
- `reports/figures/09_roc_comparison.png`
- `reports/figures/10_precision_recall_comparison.png`

## Appendix B: Verified Data Dictionary

The following table summarises the variables used in the analysis. The official UCI documentation should be cited for the complete attribute definitions.

| Variable/group | Meaning | Data role |
|---|---|---|
| `school` | Student's school (`GP` or `MS`) | Categorical predictor |
| `sex` | Student's sex | Categorical predictor |
| `age` | Student age | Numerical predictor |
| `address` | Urban or rural home address | Categorical predictor |
| `famsize`, `Pstatus` | Family size and parent cohabitation status | Categorical predictors |
| `Medu`, `Fedu` | Mother's and father's education level | Ordinal numerical predictors |
| `Mjob`, `Fjob` | Mother's and father's occupations | Categorical predictors |
| `reason`, `guardian` | Reason for choosing school and guardian | Categorical predictors |
| `traveltime` | Travel time to school | Ordinal numerical predictor |
| `studytime` | Weekly study-time category | Ordinal numerical predictor |
| `failures` | Number of previous class failures | Numerical predictor |
| `schoolsup`, `famsup` | School and family educational support | Categorical predictors |
| `paid`, `activities`, `nursery`, `higher`, `internet`, `romantic` | Participation, aspirations, access, and relationship indicators | Categorical predictors |
| `famrel`, `freetime`, `goout` | Family relationship, free time, and social activity ratings | Ordinal numerical predictors |
| `Dalc`, `Walc` | Workday and weekend alcohol-consumption ratings | Ordinal numerical predictors |
| `health` | Current health rating | Ordinal numerical predictor |
| `absences` | Number of school absences | Numerical predictor |
| `G1`, `G2` | Earlier-period grades | Excluded to support early prediction |
| `G3` | Final grade | Used only to create the target |

## Appendix C: Verified Class Percentages

The dataset contains:

- At Risk: 130 of 395 students = **32.91%**
- Pass: 265 of 395 students = **67.09%**

This imbalance explains why a model can achieve reasonable accuracy while still missing a substantial proportion of At Risk students. At Risk recall and false-negative rate are therefore reported separately.

## Appendix D: Presentation and Viva Talking Points

### Why was education selected?

Education is a practical domain where early identification can help institutions plan support. The project demonstrates a complete Data Mining workflow using a real public dataset.

### Why were G1 and G2 removed?

They are earlier examination grades and would make the model depend on information that may not be available at the intended early-support stage. `G3` was removed because it defines the target.

### Why is At Risk recall important?

At Risk recall measures the proportion of actual At Risk students identified by the model. A false negative means a student who may need support is predicted as Pass, so this error deserves special attention.

### Which model should be trusted?

No model should be described as universally best from this small experiment. Training cross-validation favoured the selected Random Forest at the chosen threshold, but its final test At Risk recall was only 0.038. This difference is an important limitation and shows why further validation is required before real deployment.

### What decision does the system support?

It supports a teacher's decision about whom to review for possible support. It does not assign grades, punish students, or replace teacher judgement.

### What would be done next?

Collect a newer local cohort, validate the model externally, examine subgroup fairness, calibrate probabilities, and involve teachers when defining an acceptable precision-recall trade-off.

