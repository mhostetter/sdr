import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Number of coherent samples must be at least 1
        sdr.coherent_gain(0)


def test_scalar():
    assert sdr.coherent_gain(1) == pytest.approx(0.0)
    assert sdr.coherent_gain(2) == pytest.approx(3.010299956639813)
    assert sdr.coherent_gain(10) == pytest.approx(10.0)
    assert sdr.coherent_gain(20) == pytest.approx(13.010299956639813)


def test_vector():
    n_c = np.array([1, 2, 10, 20])
    g_c = sdr.coherent_gain(n_c)
    g_c_truth = np.array([0.0, 3.010299956639813, 10.0, 13.010299956639813])
    assert isinstance(g_c, np.ndarray)
    assert np.allclose(g_c, g_c_truth)
