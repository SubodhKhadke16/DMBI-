from src.data_preprocessing import load_data
from src.train_models import train_models


def test_all_models_train_and_predict():
    models, X_test, _, y_test, _ = train_models(load_data())
    assert set(models) == {"Decision Tree", "Random Forest", "Logistic Regression"}
    for model in models.values():
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        assert len(predictions) == len(y_test)
        assert probabilities.shape == (len(y_test), 2)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert abs(probabilities.sum(axis=1) - 1).max() < 1e-8

