import numpy as np
import pytest

import sdr


def test_standard_limits():
    assert sdr.shannon_limit_snr(0) == -np.inf
    assert sdr.shannon_limit_snr(1) == 0
    assert sdr.shannon_limit_snr(2) == pytest.approx(4.771212547196624)


def test_vector():
    """
    These numbers were verified visually from published papers.
    """
    rate = np.array([1 / 8, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 7 / 8])  # Code rate, n/k
    rho = 2 * rate  # bits/2D
    ebn0 = sdr.shannon_limit_snr(rho)
    ebn0_truth = np.array([-7.23062536, -3.82775685, -2.3106528, 0.0, 1.8179847, 2.62077655, 3.73571347])
    assert np.allclose(ebn0, ebn0_truth)
