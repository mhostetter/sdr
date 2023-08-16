import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        x = np.array([1, 0, 0, 0, 1, 1, 0, 1], dtype=float)
        sdr.pack(x, 2)
    with pytest.raises(ValueError):
        x = np.array([1, 0, 0, 0, 1, 1, 0, 1], dtype=int)
        sdr.pack(x, 0)


def test_dtype():
    x = np.random.randint(0, 2, 50)
    assert sdr.pack(x, 7).dtype == np.uint8
    assert sdr.pack(x, 8).dtype == np.uint8
    assert sdr.pack(x, 9).dtype == np.uint16
    assert sdr.pack(x, 17).dtype == np.uint32

    assert sdr.pack(x, 7, dtype=np.uint16).dtype == np.uint16


def test_0d():
    assert np.array_equal(sdr.pack(0, 3), [0])
    assert np.array_equal(sdr.pack(1, 3), [4])


def test_1d():
    # 32 elements
    x = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    assert np.array_equal(sdr.pack(x, 1), x)
    assert np.array_equal(sdr.pack(x, 2), [2, 0, 3, 1, 3, 0, 2, 3, 0, 2, 3, 1, 1, 3, 1, 0])
    assert np.array_equal(sdr.pack(x, 3), [4, 3, 3, 4, 5, 4, 5, 5, 3, 5, 0])
    assert np.array_equal(sdr.pack(x, 4), [8, 13, 12, 11, 2, 13, 7, 4])
    assert np.array_equal(sdr.pack(x, 5), [17, 23, 5, 18, 26, 29, 0])
    assert np.array_equal(sdr.pack(x, 6), [35, 28, 44, 45, 29, 0])
    assert np.array_equal(sdr.pack(x, 7), [70, 114, 101, 87, 32])
    assert np.array_equal(sdr.pack(x, 8), [141, 203, 45, 116])
    assert np.array_equal(sdr.pack(x, 11), [1134, 715, 744])
    assert np.array_equal(sdr.pack(x, 16), [36299, 11636])


def test_2d():
    # 2 x 16 elements
    x = [[1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]]
    assert np.array_equal(sdr.pack(x, 1), x)
    assert np.array_equal(sdr.pack(x, 2), [[3, 1, 0, 3, 2, 1, 0, 3], [2, 2, 3, 3, 3, 3, 1, 2]])
    assert np.array_equal(sdr.pack(x, 3), [[6, 4, 7, 1, 1, 4], [5, 3, 7, 7, 3, 0]])
    assert np.array_equal(sdr.pack(x, 4), [[13, 3, 9, 3], [10, 15, 15, 6]])
    assert np.array_equal(sdr.pack(x, 5), [[26, 14, 9, 16], [21, 31, 27, 0]])
    assert np.array_equal(sdr.pack(x, 6), [[52, 57, 12], [43, 63, 24]])
    assert np.array_equal(sdr.pack(x, 7), [[105, 100, 96], [87, 125, 64]])
    assert np.array_equal(sdr.pack(x, 8), [[211, 147], [175, 246]])
    assert np.array_equal(sdr.pack(x, 11), [[1692, 1216], [1407, 1408]])
    assert np.array_equal(sdr.pack(x, 16), [[54163], [45046]])
