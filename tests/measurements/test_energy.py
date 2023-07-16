import numpy as np
import pytest

import sdr


def test_constant():
    A = np.random.uniform(2, 10)  # Signal amplitude
    N = 50
    x = A * np.ones(N)
    assert sdr.energy(x) == pytest.approx(A**2 * N, rel=1e-2)


def test_sine():
    A = np.random.uniform(2, 10)  # Signal amplitude
    N = 50
    x = A * np.sin(np.linspace(0, 2 * np.pi, N, endpoint=False))
    assert sdr.energy(x) == pytest.approx(A**2 / 2 * N, rel=1e-2)
