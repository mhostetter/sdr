import numpy as np

import sdr


def test_exceptions():
    with np.testing.assert_raises(TypeError):
        sdr.BlockInterleaver(3.0, 4)
    with np.testing.assert_raises(TypeError):
        sdr.BlockInterleaver(3, 4.0)


def test_simple():
    interleaver = sdr.BlockInterleaver(3, 4)
    assert len(interleaver) == 12

    map = np.array([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11])
    assert np.array_equal(interleaver.map, map)

    inverse_map = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11])
    assert np.array_equal(interleaver.inverse_map, inverse_map)

    x_truth = np.arange(24)
    y_truth = np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 12, 15, 18, 21, 13, 16, 19, 22, 14, 17, 20, 23])

    y = interleaver.interleave(x_truth)
    assert np.array_equal(y, y_truth)

    x = interleaver.deinterleave(y)
    assert np.array_equal(x, x_truth)
