import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Input must be 1D
        sdr.diff_encode([[0, 1, 0], [0, 1, 1]])
    with pytest.raises(TypeError):
        # Input must be integer
        sdr.diff_encode([0, 1.0, 0])
    with pytest.raises(TypeError):
        # y_prev must be integer
        sdr.diff_encode([0, 1, 0], y_prev=1.0)
    with pytest.raises(ValueError):
        # y_prev must be non-negative
        sdr.diff_encode([0, 1, 0], y_prev=-1)


def test_encode():
    x = [0, 1, 0, 0, 1, 1]
    y = sdr.diff_encode(x)
    y_truth = [0, 1, 1, 1, 0, 1]
    assert np.array_equal(y, y_truth)
