import numpy as np
import pytest

import sdr


def test_absolute_limit():
    esn0 = -100  # dB
    C = sdr.awgn_capacity(esn0)
    ebn0 = esn0 - 10 * np.log10(C)
    ebn0_limit = 10 * np.log10(np.log(2))  # ~ -1.59 dB
    assert ebn0 == pytest.approx(ebn0_limit)


def test_vector():
    """
    These numbers were verified visually from published papers.
    """
    snr = np.linspace(-30, 20, 11)
    C = sdr.awgn_capacity(snr)
    C_truth = np.array(
        [
            1.44197417e-03,
            4.55500399e-03,
            1.43552930e-02,
            4.49155310e-02,
            1.37503524e-01,
            3.96409161e-01,
            1.00000000e00,
            2.05737321e00,
            3.45943162e00,
            5.02780767e00,
            6.65821148e00,
        ]
    )
    assert np.allclose(C, C_truth)
