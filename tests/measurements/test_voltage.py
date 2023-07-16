import numpy as np
import pytest

import sdr


def test_peak_voltage():
    A = np.random.uniform(2, 10)  # Signal amplitude
    N = 50
    x = A * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    assert sdr.peak_voltage(x) == pytest.approx(A, rel=1e-2)


def test_rms_voltage():
    A = np.random.uniform(2, 10)  # Signal amplitude
    N = 50
    x = A * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    assert sdr.rms_voltage(x) == pytest.approx(A / np.sqrt(2), rel=1e-2)
