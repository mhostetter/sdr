import numpy as np

import sdr

X = np.array([[6, 5, 0, 0, 5], [0, 1, 5, 0, 3], [4, 3, 3, 0, 5]])
Y = np.array([[7, 5, 2, 4, 0], [0, 0, 4, 1, 5], [5, 0, 7, 5, 6]])


def test_axis_none():
    d = sdr.hamming(X, Y)
    assert d.shape == ()
    assert d == 37


def test_axis_0():
    d = sdr.hamming(X, Y, axis=0)
    assert d.shape == (5,)
    assert np.allclose(d, [2, 4, 7, 10, 14])


def test_axis_1():
    d = sdr.hamming(X, Y, axis=1)
    assert d.shape == (3,)
    assert np.allclose(d, [12, 9, 16])
