import numpy as np
import pytest

import sdr


def test_power():
    rng = np.random.default_rng()
    A = rng.uniform(2, 10)  # Signal amplitude
    N = 50
    x = A * np.exp(1j * np.linspace(0, 2 * np.pi, N, endpoint=False))
    assert np.allclose(sdr.power(x), A**2)


def test_peak_power():
    rng = np.random.default_rng()
    A = rng.uniform(2, 10)  # Signal amplitude
    N = 50
    x = A * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    assert sdr.peak_power(x) == pytest.approx(A**2, rel=1e-2)


def test_average_power():
    rng = np.random.default_rng()
    A = rng.uniform(2, 10)  # Signal amplitude
    N = 50
    x = A * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    assert sdr.average_power(x) == pytest.approx(A**2 / 2, rel=1e-2)
