import pytest

import sdr


def test_stein_1():
    snr1 = -10  # dB
    snr2 = -10  # dB
    time = 1e-3  # seconds
    bandwidth = 1e6  # Hz
    sigma_fdoa = sdr.fdoa_crlb(snr1, snr2, time, bandwidth)
    assert sigma_fdoa == pytest.approx(135.04744742356585)  # Stein says ~140 Hz


def test_stein_2():
    snr1 = -10  # dB
    snr2 = -10  # dB
    time = 6e-3  # seconds
    bandwidth = 1e6  # Hz
    sigma_fdoa = sdr.fdoa_crlb(snr1, snr2, time, bandwidth)
    assert sigma_fdoa == pytest.approx(9.188814923696532)  # Stein says ~9 ns
