import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Transition probability must be between 0 and 1
        sdr.bsc_capacity(-0.1)
    with pytest.raises(ValueError):
        # Transition probability must be between 0 and 1
        sdr.bsc_capacity(1.1)


def test_limit_cases():
    assert sdr.bsc_capacity(0) == 1
    assert sdr.bsc_capacity(1) == 1


def test_outputs():
    C = sdr.bsc_capacity([0, 0.5, 1])
    assert isinstance(C, np.ndarray)
    assert np.array_equal(C, [1, 0, 1])

    C = sdr.bsc_capacity(0.5)
    assert isinstance(C, float)
    assert C == 0
