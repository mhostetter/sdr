import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Input must be 1D
        sdr.diff_decode([[0, 1, 0], [0, 1, 1]])
    with pytest.raises(TypeError):
        # Input must be integer
        sdr.diff_decode([0, 1.0, 0])
    with pytest.raises(TypeError):
        # y_prev must be integer
        sdr.diff_decode([0, 1, 0], y_prev=1.0)
    with pytest.raises(ValueError):
        # y_prev must be non-negative
        sdr.diff_decode([0, 1, 0], y_prev=-1)


def test_decode_1():
    """
    Matlab:
        >> diff_dec = comm.DifferentialDecoder;
        >> y = randi([0 1], 10, 1); y
        >> diff_dec(y)
    """
    y = [1, 0, 1, 1, 0, 0, 0, 0, 0, 1]
    x = sdr.diff_decode(y)
    x_truth = [1, 1, 1, 0, 1, 0, 0, 0, 0, 1]
    assert np.array_equal(x, x_truth)


# TODO: Add more Matlab tests
