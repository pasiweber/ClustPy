import numpy as np
from clustpy.metrics import s_dbw_score


def test_s_dbw_score():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [5, 5], [4, 5], [5, 4], [4, 4], [100, 100], [101, 101]])
    L1 = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1])
    L2 = np.array([0, 0, 1, 1, 1, 1, 1, 1, -1, -1])
    L3 = np.array([0, 0, 1, 1, 1, 1, 0, 0, -1, -1])

    s_dbw_correct_labels = s_dbw_score(X, L1)
    s_dbw_wrong_labels1 = s_dbw_score(X, L2)
    s_dbw_wrong_labels2 = s_dbw_score(X, L3)

    assert s_dbw_correct_labels < s_dbw_wrong_labels1
    assert s_dbw_correct_labels < s_dbw_wrong_labels2
    assert s_dbw_wrong_labels1 < s_dbw_wrong_labels2
