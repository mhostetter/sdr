import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        x = np.array([8, 13, 12], dtype=float)
        sdr.pack(x, 4)
    with pytest.raises(ValueError):
        x = np.array([8, 13, 12], dtype=int)
        sdr.pack(x, 0)


def test_dtype():
    x = np.random.randint(0, 2**7, 5, dtype=int)
    assert sdr.unpack(x, 7).dtype == np.uint8
    x = np.random.randint(0, 2**17, 5, dtype=int)
    assert sdr.unpack(x, 17).dtype == np.uint8

    assert sdr.unpack(x, 7, dtype=np.uint16).dtype == np.uint16


def test_0d():
    assert np.array_equal(sdr.unpack(5, 3), [1, 0, 1])
    assert np.array_equal(sdr.unpack(2, 3), [0, 1, 0])


def test_1d():
    # 32 elements
    x = [1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
    assert np.array_equal(sdr.unpack(x, 1), x)
    assert np.array_equal(sdr.unpack([2, 0, 3, 1, 3, 0, 2, 3, 0, 2, 3, 1, 1, 3, 1, 0], 2), x)
    assert np.array_equal(sdr.unpack([4, 3, 3, 4, 5, 4, 5, 5, 3, 5, 0], 3), x + [0])
    assert np.array_equal(sdr.unpack([8, 13, 12, 11, 2, 13, 7, 4], 4), x)
    assert np.array_equal(sdr.unpack([17, 23, 5, 18, 26, 29, 0], 5), x + [0] * 3)
    assert np.array_equal(sdr.unpack([35, 28, 44, 45, 29, 0], 6), x + [0] * 4)
    assert np.array_equal(sdr.unpack([70, 114, 101, 87, 32], 7), x + [0] * 3)
    assert np.array_equal(sdr.unpack([141, 203, 45, 116], 8), x)
    assert np.array_equal(sdr.unpack([1134, 715, 744], 11), x + [0])
    assert np.array_equal(sdr.unpack([36299, 11636], 16), x)


def test_2d():
    # 2 x 16 elements
    x1 = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1]
    x2 = [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
    x = [x1, x2]
    assert np.array_equal(sdr.unpack(x, 1), x)
    assert np.array_equal(sdr.unpack([[3, 1, 0, 3, 2, 1, 0, 3], [2, 2, 3, 3, 3, 3, 1, 2]], 2), x)
    assert np.array_equal(sdr.unpack([[6, 4, 7, 1, 1, 4], [5, 3, 7, 7, 3, 0]], 3), [x1 + [0] * 2, x2 + [0] * 2])
    assert np.array_equal(sdr.unpack([[13, 3, 9, 3], [10, 15, 15, 6]], 4), x)
    assert np.array_equal(sdr.unpack([[26, 14, 9, 16], [21, 31, 27, 0]], 5), [x1 + [0] * 4, x2 + [0] * 4])
    assert np.array_equal(sdr.unpack([[52, 57, 12], [43, 63, 24]], 6), [x1 + [0] * 2, x2 + [0] * 2])
    assert np.array_equal(sdr.unpack([[105, 100, 96], [87, 125, 64]], 7), [x1 + [0] * 5, x2 + [0] * 5])
    assert np.array_equal(sdr.unpack([[211, 147], [175, 246]], 8), x)
    assert np.array_equal(sdr.unpack([[1692, 1216], [1407, 1408]], 11), [x1 + [0] * 6, x2 + [0] * 6])
    assert np.array_equal(sdr.unpack([[54163], [45046]], 16), x)
