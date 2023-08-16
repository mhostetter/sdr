import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(ValueError):
        # Need either 'snr' or 'noise'
        sdr.awgn(np.ones(10))


def test_real_from_snr():
    A = np.random.uniform(2, 3)  # Signal amplitude
    N = 1000
    x = A * np.ones(N, dtype=float)  # Use a constant signal so the power measurement has no uncertainty

    snr = np.random.uniform(1, 2)  # dB
    snr_linear = 10 ** (snr / 10)
    signal_power = A**2  # Theoretical power of a constant signal
    noise_std = np.sqrt(signal_power / snr_linear)

    y = sdr.awgn(x, snr=snr)

    # Use standard deviation so the measurement errors aren't compounded
    std = np.std(y - x)
    assert std == pytest.approx(noise_std, rel=1e-1)


def test_complex_from_snr():
    A = np.random.uniform(2, 3)  # Signal amplitude
    N = 1000
    x = A * np.ones(N, dtype=complex)  # Use a constant signal so the power measurement has no uncertainty

    snr = np.random.uniform(1, 2)  # dB
    snr_linear = 10 ** (snr / 10)
    signal_power = A**2  # Theoretical power of a constant signal
    noise_std = np.sqrt(signal_power / snr_linear)

    y = sdr.awgn(x, snr=snr)

    # Use standard deviation so the measurement errors aren't compounded
    std = np.sqrt(np.var((y - x).real) + np.var((y - x).imag))
    assert std == pytest.approx(noise_std, rel=1e-1)


def test_real_from_noise():
    A = np.random.uniform(2, 3)  # Signal amplitude
    N = 1000
    x = A * np.ones(N, dtype=float)  # Use a constant signal so the power measurement has no uncertainty

    noise_std = np.random.uniform(1, 2)
    y = sdr.awgn(x, noise=noise_std**2)

    # Use standard deviation so the measurement errors aren't compounded
    std = np.std(y - x)
    assert std == pytest.approx(noise_std, rel=1e-1)


def test_complex_from_noise():
    A = np.random.uniform(2, 3)  # Signal amplitude
    N = 1000
    x = A * np.ones(N, dtype=complex)  # Use a constant signal so the power measurement has no uncertainty

    noise_std = np.random.uniform(1, 2)
    y = sdr.awgn(x, noise=noise_std**2)

    # Use standard deviation so the measurement errors aren't compounded
    std = np.sqrt(np.var((y - x).real) + np.var((y - x).imag))
    assert std == pytest.approx(noise_std, rel=1e-1)


def test_seed():
    x = np.ones(100)
    y1 = sdr.awgn(x, snr=10, seed=0)
    y2 = sdr.awgn(x, snr=10, seed=0)
    assert np.all(y1 == y2)
