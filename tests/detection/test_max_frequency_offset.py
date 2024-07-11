import numpy as np
import pytest

import sdr


def test_scalar():
    assert sdr.max_frequency_offset(3, 1e-3) == pytest.approx(442.2433896262681)


def test_cgl_vector():
    f = sdr.max_frequency_offset([1, 2, 3, 4, 5], 1e3)
    f_truth = np.array([0.0002615, 0.00036547, 0.00044224, 0.00050444, 0.00055699])
    assert isinstance(f, np.ndarray)
    assert np.allclose(f, f_truth)


def test_freq_vector():
    f = sdr.max_frequency_offset(3, [1e-3, 2e-3, 3e-3, 4e-3, 5e-3])
    f_truth = np.array([442.24338963, 221.12169481, 147.41446321, 110.56084741, 88.44867793])
    assert isinstance(f, np.ndarray)
    assert np.allclose(f, f_truth)
