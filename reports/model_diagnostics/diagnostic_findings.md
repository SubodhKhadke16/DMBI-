# Model Threshold Diagnostics

## Verified probability semantics

- The fitted classifiers expose classes in the order `[0, 1]`.
- Class `0` is At Risk and class `1` is Pass.
- `predict_proba()[:, 0]` is therefore the At Risk probability.
- The corrected rule is: predict At Risk when At Risk probability is at least the threshold; otherwise predict Pass.

## Historical threshold issue

The historical implementation created a thresholded array with `1` for At Risk and then passed it to metric functions with `pos_label=0`, while the original target still used `0` for At Risk. That mixed encoding reverses the intended At Risk interpretation in the saved historical threshold table. The original files were preserved. This diagnostics run uses a consistent temporary risk encoding (`1 = At Risk`) and does not overwrite the historical results.

## Corrected training-only threshold findings

Corrected threshold metrics are in `oof_threshold_metrics.csv`. They use only training out-of-fold probabilities. With At Risk recall as the primary objective, the lowest tested threshold (0.30) gives the Random Forest an out-of-fold At Risk recall of 0.933, compared with 0.048 at threshold 0.70. This is a precision-recall trade-off, not a guarantee of generalisation.

## Fold stability

Per-fold choices are in `fold_threshold_stability.csv`. Random Forest selected 0.30 in all five folds, while Decision Tree and Logistic Regression showed some fold variation. This indicates threshold stability is model-dependent and should be reported rather than assumed.

## Held-out test findings

The corrected selected configurations were frozen after training-only decisions and then evaluated on the test set. Results are in `frozen_test_diagnostics.csv`. The test partition contains only 26 At Risk students, so one changed prediction changes recall by approximately 0.038. Wilson intervals are in `test_recall_uncertainty.csv` and illustrate the resulting uncertainty.

## Recommendation

For the stated academic objective, use the corrected At Risk-probability rule and do not use the historical 0.70 selection result without correction. If recall is the only priority among the tested thresholds, 0.30 is the transparent choice in this experiment; however, it generates many referrals. A teacher/institution should choose an operating threshold using a pre-declared acceptable false-positive and support capacity policy, then validate it on a newer cohort.

## Calibration

Brier scores and calibration points are saved in `frozen_test_diagnostics.csv` and `calibration_points.csv`. Calibration curves should be interpreted cautiously because the held-out test contains only 79 students and 26 At Risk cases.
