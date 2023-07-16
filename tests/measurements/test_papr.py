"""
References:
    - https://en.wikipedia.org/wiki/Crest_factor
"""
import numpy as np
import pytest

import sdr


def test_sine():
    N = 50
    x = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    assert sdr.papr(x) == pytest.approx(3.0, rel=1e-2)


def test_rectified_sine():
    N = 50
    x = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    x = np.abs(x)
    assert sdr.papr(x) == pytest.approx(3.0, rel=1e-2)


def test_halfwave_sine():
    N = 50
    x = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    x[N // 2 :] = 0
    assert sdr.papr(x) == pytest.approx(6.0, rel=1e-2)


def test_square_wave():
    N = 50
    x = np.ones(N)
    x[N // 2 :] = -1
    assert sdr.papr(x) == pytest.approx(0.0, rel=1e-2)


def test_pwm():
    n1 = 10
    N = 50
    x = np.zeros(N)
    x[0:n1] = 1
    assert sdr.papr(x) == pytest.approx(10 * np.log10(N / n1), rel=1e-2)
