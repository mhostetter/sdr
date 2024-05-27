import pytest

import sdr


def test_stein_1():
    # Modified for TOA from Stein's example
    snr1 = -10  # dB
    time = 1e-3  # seconds
    bandwidth = 1e6  # Hz
    sigma_tdoa = sdr.toa_crlb(snr1, time, bandwidth)
    assert sigma_tdoa == pytest.approx(38.9848400616838e-9)


def test_stein_2():
    # Modified for TOA from Stein's example
    snr1 = -10  # dB
    time = 6e-3  # seconds
    bandwidth = 1e6  # Hz
    sigma_tdoa = sdr.toa_crlb(snr1, time, bandwidth)
    assert sigma_tdoa == pytest.approx(15.915494309189534e-9)
