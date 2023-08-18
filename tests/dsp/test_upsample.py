import numpy as np
import pytest

import sdr


def test_exceptions():
    x = np.random.randn(10)

    with pytest.raises(TypeError):
        # Rate must be an integer
        sdr.upsample(x, 4.0)
    with pytest.raises(ValueError):
        # Rate must be positive
        sdr.upsample(x, 0)


def test_1():
    """
    Matlab:
        >> x = 0:9;
        >> upsample(x, 4)'
    """
    x = np.arange(10)
    y = sdr.upsample(x, 4)
    y_truth = np.array(
        [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            5,
            0,
            0,
            0,
            6,
            0,
            0,
            0,
            7,
            0,
            0,
            0,
            8,
            0,
            0,
            0,
            9,
            0,
            0,
            0,
        ]
    )
    assert np.array_equal(y, y_truth)
