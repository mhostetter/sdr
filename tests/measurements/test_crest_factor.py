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
    assert sdr.crest_factor(x) == pytest.approx(np.sqrt(2), rel=1e-1)


def test_rectified_sine():
    N = 50
    x = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    x = np.abs(x)
    assert sdr.crest_factor(x) == pytest.approx(np.sqrt(2), rel=1e-1)


def test_halfwave_sine():
    N = 50
    x = np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    x[N // 2 :] = 0
    assert sdr.crest_factor(x) == pytest.approx(2.0, rel=1e-1)


def test_square_wave():
    N = 50
    x = np.ones(N)
    x[N // 2 :] = -1
    assert sdr.crest_factor(x) == pytest.approx(1.0, rel=1e-1)


def test_pwm():
    n1 = 10
    N = 50
    x = np.zeros(N)
    x[0:n1] = 1
    assert sdr.crest_factor(x) == pytest.approx(np.sqrt(N / n1), rel=1e-1)
