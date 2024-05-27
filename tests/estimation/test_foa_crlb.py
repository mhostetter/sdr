import pytest

import sdr


def test_stein_1():
    # Modified for FOA from Stein's example
    snr1 = -10  # dB
    time = 1e-3  # seconds
    bandwidth = 1e6  # Hz
    sigma_foa = sdr.foa_crlb(snr1, time, bandwidth)
    assert sigma_foa == pytest.approx(38.9848400616838)


def test_stein_2():
    # Modified for FOA from Stein's example
    snr1 = -10  # dB
    time = 6e-3  # seconds
    bandwidth = 1e6  # Hz
    sigma_foa = sdr.foa_crlb(snr1, time, bandwidth)
    assert sigma_foa == pytest.approx(2.652582384864922)
