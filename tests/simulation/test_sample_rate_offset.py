import numpy as np

import sdr


def expected_length(n: int, rate: float) -> int:
    return int(np.ceil(n * rate - 1e-12))


def test_sample_rate_offset_zero():
    x = np.arange(100, dtype=float)

    y = sdr.sample_rate_offset(x, 0)

    assert y.size == x.size
    np.testing.assert_allclose(y, x)


def test_sample_rate_offset_positive_linear_ramp():
    x = np.arange(100, dtype=float)
    offset = 0.25
    rate = 1 + offset

    y = sdr.sample_rate_offset(x, offset)

    expected = np.arange(y.size) / rate

    assert y.size == expected_length(x.size, rate)
    np.testing.assert_allclose(y[5:-5], expected[5:-5], atol=1e-10)


def test_sample_rate_offset_negative_linear_ramp():
    x = np.arange(100, dtype=float)
    offset = -0.25
    rate = 1 + offset

    y = sdr.sample_rate_offset(x, offset)

    expected = np.arange(y.size) / rate

    assert y.size == expected_length(x.size, rate)
    np.testing.assert_allclose(y[5:-5], expected[5:-5], atol=1e-10)


def test_sample_rate_offset_hz_linear_ramp():
    sample_rate = 1_000.0
    x = np.arange(100, dtype=float)
    offset = 100.0
    rate = (sample_rate + offset) / sample_rate

    y = sdr.sample_rate_offset(x, offset, sample_rate=sample_rate)

    expected = np.arange(y.size) / rate

    assert y.size == expected_length(x.size, rate)
    np.testing.assert_allclose(y[5:-5], expected[5:-5], atol=1e-10)


def test_sample_rate_offset_array_constant_linear_ramp():
    x = np.arange(100, dtype=float)
    offset = np.full(x.size, 0.25)
    rate = 1.25

    y = sdr.sample_rate_offset(x, offset)

    expected = np.arange(y.size) / rate

    assert y.size == expected_length(x.size, rate)
    np.testing.assert_allclose(y[5:-5], expected[5:-5], atol=1e-10)


def test_sample_rate_offset_inverse_linear_ramp():
    x = np.arange(200, dtype=float)
    offset = 0.25

    y = sdr.sample_rate_offset(x, offset)
    z = sdr.sample_rate_offset(y, -offset / (1 + offset))

    n = min(x.size, z.size)

    np.testing.assert_allclose(z[10 : n - 10], x[10 : n - 10], atol=1e-8)


def test_sample_rate_offset_complex_sinusoid_inverse():
    n = np.arange(1_000)
    x = np.exp(1j * 2 * np.pi * 0.03 * n)
    offset = 0.01

    y = sdr.sample_rate_offset(x, offset)
    z = sdr.sample_rate_offset(y, -offset / (1 + offset))

    n = min(x.size, z.size)

    np.testing.assert_allclose(z[50 : n - 50], x[50 : n - 50], atol=2e-3, rtol=2e-3)


def test_sample_rate_offset_rate_linear_ramp():
    x = np.arange(200, dtype=float)
    offset = 0.02
    offset_rate = -0.01 / x.size

    y = sdr.sample_rate_offset(x, offset, offset_rate=offset_rate)

    assert y.size > 0
    assert np.all(np.isfinite(y))
    assert y[20] > x[20] / (1 + offset)


def test_sample_rate_offset_real_output_for_real_input():
    x = np.arange(100, dtype=float)

    y = sdr.sample_rate_offset(x, 0.25)

    assert not np.iscomplexobj(y)


def test_sample_rate_offset_complex_output_for_complex_input():
    x = np.exp(1j * 2 * np.pi * 0.03 * np.arange(100))

    y = sdr.sample_rate_offset(x, 0.25)

    assert np.iscomplexobj(y)
