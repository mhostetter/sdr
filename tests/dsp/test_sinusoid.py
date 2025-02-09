import numpy as np

import sdr


def test_complex():
    sample_rate = 1e3  # samples/s
    N = 100  # samples
    freq = 10  # Hz
    phase = 45  # degrees
    lo = sdr.sinusoid(N / sample_rate, freq=freq, phase=phase, sample_rate=sample_rate, complex=True)
    lo_truth = np.exp(1j * (2 * np.pi * freq / sample_rate * np.arange(N) + np.deg2rad(phase)))
    assert np.allclose(lo, lo_truth)


def test_real():
    sample_rate = 1e3  # samples/s
    N = 100  # samples
    freq = 10  # Hz
    phase = 45  # degrees
    lo = sdr.sinusoid(N / sample_rate, freq=freq, phase=phase, sample_rate=sample_rate, complex=False)
    lo_truth = np.cos(2 * np.pi * freq / sample_rate * np.arange(N) + np.deg2rad(phase))
    assert np.allclose(lo, lo_truth)
