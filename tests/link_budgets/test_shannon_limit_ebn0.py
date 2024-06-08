import numpy as np
import pytest

import sdr


def test_absolute_limit():
    assert sdr.shannon_limit_ebn0(0) == pytest.approx(-1.591745389548616)


def test_vector():
    """
    These numbers were verified visually from published papers.
    """
    rate = np.array([1 / 8, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 7 / 8])  # Code rate, n/k
    rho = 2 * rate  # bits/2D
    ebn0 = sdr.shannon_limit_ebn0(rho)
    ebn0_truth = np.array(
        [
            -1.21002545,
            -0.8174569,
            -0.54974021,  # Verified from a paper
            0.0,
            0.56859734,
            0.85986396,
            1.30533298,
        ],
    )
    assert np.allclose(ebn0, ebn0_truth)
