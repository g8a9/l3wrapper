from l3wrapper.l3wrapper import create_column_names
import numpy as np

def test_create_column_names():
    X = np.zeros((19,4))
    cn = create_column_names(X)
    assert len(cn) == X.shape[1]
    assert cn == ['1', '2', '3', '4']