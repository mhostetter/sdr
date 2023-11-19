import numpy as np
import pytest

import sdr

X = np.array([[6, 5, 0, 0, 5], [0, 1, 5, 0, 3], [4, 3, 3, 0, 5]])
Y = np.array([[7, 5, 2, 4, 0], [0, 0, 4, 1, 5], [5, 0, 7, 5, 6]])


def test_axis_none():
    d = sdr.euclidean(X, Y)
    assert d.shape == ()
    assert d == pytest.approx(10.246950765959598)


def test_axis_0():
    d = sdr.euclidean(X, Y, axis=0)
    assert d.shape == (5,)
    assert np.allclose(d, [1.41421356, 3.16227766, 4.58257569, 6.4807407, 5.47722558])


def test_axis_1():
    d = sdr.euclidean(X, Y, axis=1)
    assert d.shape == (3,)
    assert np.allclose(d, [6.78232998, 2.64575131, 7.21110255])
