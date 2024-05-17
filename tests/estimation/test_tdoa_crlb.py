import sdr


def test_stein_1():
    snr1 = -10  # dB
    snr2 = -10  # dB
    time = 1e-3  # seconds
    bandwidth = 1e6  # Hz
    sigma_tdoa = sdr.tdoa_crlb(snr1, snr2, time, bandwidth)
    assert sigma_tdoa == 135.04744742356585e-9  # Stein says ~140 ns


def test_stein_2():
    snr1 = -10  # dB
    snr2 = -10  # dB
    time = 6e-3  # seconds
    bandwidth = 1e6  # Hz
    sigma_tdoa = sdr.tdoa_crlb(snr1, snr2, time, bandwidth)
    assert sigma_tdoa == 55.132889542179175e-9  # Stein says ~56 ns
