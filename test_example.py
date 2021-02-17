import numpy as np

def test_type_data():
    data = np.genfromtxt("data/train_features.csv")
    assert True == isinstance(data, np.ndarray) 

def test_missing_data():
    data = np.genfromtxt("data/train_features.csv")
    assert 0 == np.isnan(data).sum()
