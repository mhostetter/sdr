import numpy as np

import sdr


def test_1():
    """
    MATLAB:
        >> x = 0:39;
        >> downsample(x, 4)'
    """
    x = np.arange(40)
    y = sdr.downsample(x, 4)
    y_truth = np.array([0, 4, 8, 12, 16, 20, 24, 28, 32, 36])
    assert np.array_equal(y, y_truth)
