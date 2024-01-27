import numpy as np

import sdr


def test_exceptions():
    with np.testing.assert_raises(TypeError):
        sdr.Interleaver(0)
    with np.testing.assert_raises(TypeError):
        # Values must be integers
        sdr.Interleaver(np.array([0, 1.0]))
    with np.testing.assert_raises(ValueError):
        # Values must be exhaustive in [0, len(map))
        sdr.Interleaver(np.array([0, 1, 5, 2]))
    with np.testing.assert_raises(ValueError):
        # Values must be unique
        sdr.Interleaver(np.array([0, 1, 2, 2]))


def test_simple():
    map = np.array([0, 3, 1, 2])
    inverse_map = np.array([0, 2, 3, 1])
    interleaver = sdr.Interleaver(map)

    assert len(interleaver) == 4
    assert np.array_equal(interleaver.map, map)
    assert np.array_equal(interleaver.inverse_map, inverse_map)

    x_truth = np.arange(16)
    y_truth = np.array([0, 2, 3, 1, 4, 6, 7, 5, 8, 10, 11, 9, 12, 14, 15, 13])

    y = interleaver.interleave(x_truth)
    assert np.array_equal(y, y_truth)

    x = interleaver.deinterleave(y)
    assert np.array_equal(x, x_truth)
