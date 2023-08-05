import numpy as np
import pytest

import sdr


def test_exceptions():
    x = np.random.randn(10)

    with pytest.raises(TypeError):
        sdr.mix(x, freq="10")
    with pytest.raises(TypeError):
        sdr.mix(x, phase="45")
    with pytest.raises(TypeError):
        sdr.mix(x, sample_rate="1e3")

    with pytest.raises(ValueError):
        sdr.mix(x, freq=0.6)
    with pytest.raises(ValueError):
        sdr.mix(x, freq=5e3 + 1, sample_rate=10e3)
    with pytest.raises(ValueError):
        sdr.mix(x, sample_rate=0)


def test_freq():
    sample_rate = 1e3  # samples/s
    freq = 10  # Hz
    N = 100  # samples
    x = np.exp(1j * (2 * np.pi * freq / sample_rate * np.arange(N)))
    y = sdr.mix(x, freq=-freq, sample_rate=sample_rate)
    assert np.allclose(y, 1)


def test_phase():
    sample_rate = 1e3  # samples/s
    phase = 45  # degrees
    N = 100  # samples
    x = np.exp(1j * np.ones(N) * np.deg2rad(phase))
    y = sdr.mix(x, phase=-phase, sample_rate=sample_rate)
    assert np.allclose(y, 1)


def test_freq_phase():
    sample_rate = 1e3  # samples/s
    freq = 10  # Hz
    phase = 45  # degrees
    N = 100  # samples
    x = np.exp(1j * (2 * np.pi * freq / sample_rate * np.arange(N) + np.deg2rad(phase)))
    y = sdr.mix(x, freq=-freq, phase=-phase, sample_rate=sample_rate)
    assert np.allclose(y, 1)
