import numpy as np

import sdr


def test_frequency_offset_zero():
    x = np.exp(1j * 2 * np.pi * 0.03 * np.arange(100))

    y = sdr.frequency_offset(x, 0)

    np.testing.assert_allclose(y, x)


def test_frequency_offset_constant_normalized():
    n = np.arange(100)
    x = np.ones(n.size, dtype=complex)

    offset = 0.05  # cycles/sample, since sample_rate defaults to 1
    y = sdr.frequency_offset(x, offset)

    expected = np.exp(1j * 2 * np.pi * offset * n)

    np.testing.assert_allclose(y, expected)


def test_frequency_offset_constant_hz():
    sample_rate = 1_000.0
    n = np.arange(100)
    t = n / sample_rate
    x = np.ones(n.size, dtype=complex)

    offset = 25.0  # Hz
    y = sdr.frequency_offset(x, offset, sample_rate=sample_rate)

    expected = np.exp(1j * 2 * np.pi * offset * t)

    np.testing.assert_allclose(y, expected)


def test_frequency_offset_adds_to_existing_frequency():
    n = np.arange(100)
    freq = 0.03
    offset = 0.05

    x = np.exp(1j * 2 * np.pi * freq * n)
    y = sdr.frequency_offset(x, offset)

    expected = np.exp(1j * 2 * np.pi * (freq + offset) * n)

    np.testing.assert_allclose(y, expected)


def test_frequency_offset_phase():
    x = np.ones(10, dtype=complex)

    y = sdr.frequency_offset(x, 0, phase=45)

    expected = np.exp(1j * np.deg2rad(45)) * x

    np.testing.assert_allclose(y, expected)


def test_frequency_offset_rate():
    n = np.arange(100)
    t = n  # sample_rate defaults to 1
    x = np.ones(n.size, dtype=complex)

    offset = 0.02
    offset_rate = 0.001
    y = sdr.frequency_offset(x, offset, offset_rate)

    expected_phase = 2 * np.pi * (offset * t + 0.5 * offset_rate * t**2)
    expected = np.exp(1j * expected_phase)

    np.testing.assert_allclose(y, expected)


def test_frequency_offset_rate_hz():
    sample_rate = 1_000.0
    n = np.arange(100)
    t = n / sample_rate
    x = np.ones(n.size, dtype=complex)

    offset = 25.0  # Hz
    offset_rate = 1_000.0  # Hz/s
    y = sdr.frequency_offset(x, offset, offset_rate, sample_rate=sample_rate)

    expected_phase = 2 * np.pi * (offset * t + 0.5 * offset_rate * t**2)
    expected = np.exp(1j * expected_phase)

    np.testing.assert_allclose(y, expected)


def test_frequency_offset_array():
    sample_rate = 1_000.0
    n = np.arange(100)
    t = n / sample_rate
    x = np.ones(n.size, dtype=complex)

    offset = np.full(n.size, 25.0)
    y = sdr.frequency_offset(x, offset, sample_rate=sample_rate)

    expected = np.exp(1j * 2 * np.pi * 25.0 * t)

    np.testing.assert_allclose(y, expected)


def test_frequency_offset_array_with_rate():
    sample_rate = 1_000.0
    n = np.arange(100)
    t = n / sample_rate
    x = np.ones(n.size, dtype=complex)

    offset = np.full(n.size, 25.0)
    offset_rate = 1_000.0
    y = sdr.frequency_offset(x, offset, offset_rate, sample_rate=sample_rate)

    expected_phase = 2 * np.pi * (offset * t + 0.5 * offset_rate * t**2)
    expected = np.exp(1j * expected_phase)

    np.testing.assert_allclose(y, expected)


def test_frequency_offset_inverse():
    n = np.arange(100)
    x = np.exp(1j * 2 * np.pi * 0.07 * n)

    y = sdr.frequency_offset(x, 0.03)
    z = sdr.frequency_offset(y, -0.03)

    np.testing.assert_allclose(z, x, atol=1e-14)


def test_frequency_offset_real_input_becomes_complex():
    x = np.ones(10)

    y = sdr.frequency_offset(x, 0.1)

    assert np.iscomplexobj(y)
    np.testing.assert_allclose(y, np.exp(1j * 2 * np.pi * 0.1 * np.arange(10)))
