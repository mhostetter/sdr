import numpy as np
import pytest

import sdr


def test_exceptions():
    x = np.random.randn(40)

    with pytest.raises(TypeError):
        # Rate must be an integer
        sdr.downsample(x, 4.0)
    with pytest.raises(ValueError):
        # Rate must be positive
        sdr.downsample(x, 0)


def test_1():
    """
    Matlab:
        >> x = 0:39;
        >> downsample(x, 4)'
    """
    x = np.arange(40)
    y = sdr.downsample(x, 4)
    y_truth = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    assert np.array_equal(y, y_truth)
