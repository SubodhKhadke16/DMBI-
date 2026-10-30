from src.data_preprocessing import load_data, prepare_features
from src.predict_student import predict_student


def test_saved_model_predicts_valid_probabilities():
    X, _ = prepare_features(load_data())
    result, probability_pass, probability_risk = predict_student(X.iloc[0].to_dict())
    assert result in {"Pass", "At Risk"}
    assert 0 <= probability_pass <= 1
    assert 0 <= probability_risk <= 1
    assert abs(probability_pass + probability_risk - 1) < 1e-8
