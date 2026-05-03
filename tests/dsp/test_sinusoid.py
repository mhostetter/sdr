import numpy as np
import pytest

import sdr


def test_sinusoid_default():
    x = sdr.sinusoid(10)

    expected = np.ones(10, dtype=complex)

    assert np.iscomplexobj(x)
    np.testing.assert_allclose(x, expected)


def test_sinusoid_length_from_duration_and_sample_rate():
    duration = 0.1
    sample_rate = 1_000.0

    x = sdr.sinusoid(duration, sample_rate=sample_rate)

    assert x.size == 100


def test_sinusoid_zero_duration():
    x = sdr.sinusoid(0)

    assert x.size == 0
    assert np.iscomplexobj(x)


def test_sinusoid_complex_frequency_normalized():
    n = np.arange(100)
    freq = 0.05

    x = sdr.sinusoid(100, freq=freq)

    expected = np.exp(1j * 2 * np.pi * freq * n)

    assert np.iscomplexobj(x)
    np.testing.assert_allclose(x, expected)


def test_sinusoid_complex_frequency_hz():
    sample_rate = 1_000.0
    duration = 0.1
    freq = 25.0

    n = np.arange(int(duration * sample_rate))
    t = n / sample_rate

    x = sdr.sinusoid(duration, freq=freq, sample_rate=sample_rate)

    expected = np.exp(1j * 2 * np.pi * freq * t)

    assert np.iscomplexobj(x)
    np.testing.assert_allclose(x, expected)


def test_sinusoid_real_frequency_normalized():
    n = np.arange(100)
    freq = 0.05

    x = sdr.sinusoid(100, freq=freq, complex=False)

    expected = np.cos(2 * np.pi * freq * n)

    assert not np.iscomplexobj(x)
    np.testing.assert_allclose(x, expected)


def test_sinusoid_real_frequency_hz():
    sample_rate = 1_000.0
    duration = 0.1
    freq = 25.0

    n = np.arange(int(duration * sample_rate))
    t = n / sample_rate

    x = sdr.sinusoid(duration, freq=freq, sample_rate=sample_rate, complex=False)

    expected = np.cos(2 * np.pi * freq * t)

    assert not np.iscomplexobj(x)
    np.testing.assert_allclose(x, expected, atol=1e-15)


def test_sinusoid_phase_complex():
    x = sdr.sinusoid(1, phase=45)

    expected = np.array([np.exp(1j * np.deg2rad(45))])

    np.testing.assert_allclose(x, expected)


def test_sinusoid_phase_real():
    x = sdr.sinusoid(1, phase=60, complex=False)

    expected = np.array([np.cos(np.deg2rad(60))])

    np.testing.assert_allclose(x, expected)


def test_sinusoid_negative_frequency():
    n = np.arange(100)
    freq = -0.05

    x = sdr.sinusoid(100, freq=freq)

    expected = np.exp(1j * 2 * np.pi * freq * n)

    np.testing.assert_allclose(x, expected)


def test_sinusoid_nyquist_positive():
    n = np.arange(10)
    freq = 0.5

    x = sdr.sinusoid(10, freq=freq)

    expected = np.exp(1j * np.pi * n)

    np.testing.assert_allclose(x, expected)


def test_sinusoid_nyquist_negative():
    n = np.arange(10)
    freq = -0.5

    x = sdr.sinusoid(10, freq=freq)

    expected = np.exp(-1j * np.pi * n)

    np.testing.assert_allclose(x, expected)


def test_sinusoid_frequency_rate_complex():
    n = np.arange(100)
    t = n  # sample_rate defaults to 1

    freq = 0.01
    freq_rate = 0.001

    x = sdr.sinusoid(100, freq=freq, freq_rate=freq_rate)

    phase_rad = 2 * np.pi * (freq * t + 0.5 * freq_rate * t**2)
    expected = np.exp(1j * phase_rad)

    np.testing.assert_allclose(x, expected)


def test_sinusoid_frequency_rate_real():
    n = np.arange(100)
    t = n  # sample_rate defaults to 1

    freq = 0.01
    freq_rate = 0.001

    x = sdr.sinusoid(100, freq=freq, freq_rate=freq_rate, complex=False)

    phase_rad = 2 * np.pi * (freq * t + 0.5 * freq_rate * t**2)
    expected = np.cos(phase_rad)

    np.testing.assert_allclose(x, expected)


def test_sinusoid_frequency_rate_hz():
    sample_rate = 1_000.0
    duration = 0.1
    freq = 10.0
    freq_rate = 100.0

    n = np.arange(int(duration * sample_rate))
    t = n / sample_rate

    x = sdr.sinusoid(duration, freq=freq, freq_rate=freq_rate, sample_rate=sample_rate)

    phase_rad = 2 * np.pi * (freq * t + 0.5 * freq_rate * t**2)
    expected = np.exp(1j * phase_rad)

    np.testing.assert_allclose(x, expected)


def test_sinusoid_frequency_phase_and_rate():
    n = np.arange(100)
    t = n

    freq = 0.01
    freq_rate = 0.001
    phase = 45

    x = sdr.sinusoid(100, freq=freq, freq_rate=freq_rate, phase=phase)

    phase_rad = 2 * np.pi * (freq * t + 0.5 * freq_rate * t**2) + np.deg2rad(phase)
    expected = np.exp(1j * phase_rad)

    np.testing.assert_allclose(x, expected)


def test_sinusoid_real_is_real_part_of_complex():
    x_complex = sdr.sinusoid(100, freq=0.05, phase=30)
    x_real = sdr.sinusoid(100, freq=0.05, phase=30, complex=False)

    np.testing.assert_allclose(x_real, x_complex.real)


def test_sinusoid_raises_for_frequency_above_nyquist():
    with pytest.raises(ValueError):
        sdr.sinusoid(10, freq=0.5001)


def test_sinusoid_raises_for_frequency_below_negative_nyquist():
    with pytest.raises(ValueError):
        sdr.sinusoid(10, freq=-0.5001)


def test_sinusoid_raises_for_negative_duration():
    with pytest.raises(ValueError):
        sdr.sinusoid(-1)


def test_sinusoid_raises_for_zero_sample_rate():
    with pytest.raises(ValueError):
        sdr.sinusoid(10, sample_rate=0)


def test_sinusoid_raises_for_negative_sample_rate():
    with pytest.raises(ValueError):
        sdr.sinusoid(10, sample_rate=-1)
