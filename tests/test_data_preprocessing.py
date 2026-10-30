from src.data_preprocessing import load_data, prepare_features


def test_dataset_shape_and_target():
    data = load_data()
    assert data.shape == (395, 33)
    X, y = prepare_features(data)
    assert set(y.unique()) == {0, 1}
    assert not {"G1", "G2", "G3"}.intersection(X.columns)


def test_dataset_has_no_missing_or_duplicate_rows():
    data = load_data()
    assert int(data.isna().sum().sum()) == 0
    assert int(data.duplicated().sum()) == 0

