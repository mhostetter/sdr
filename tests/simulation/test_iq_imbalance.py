import numpy as np
import pytest

import sdr


def test_amplitude():
    x = np.exp(1j * 2 * np.pi * 5 * np.arange(100) / 100)
    amplitude = 5  # dB
    y = sdr.iq_imbalance(x, amplitude)

    before = 10 * np.log10(sdr.average_power(x.real) / sdr.average_power(x.imag))
    after = 10 * np.log10(sdr.average_power(y.real) / sdr.average_power(y.imag))
    assert amplitude == pytest.approx(after - before)


def test_amplitude_2():
    x = np.exp(1j * 2 * np.pi * 5 * np.arange(100) / 100)
    amplitude = 5  # dB

    y = sdr.iq_imbalance(x, amplitude)

    ratio_before = 10 * np.log10(sdr.average_power(x.real) / sdr.average_power(x.imag))
    ratio_after = 10 * np.log10(sdr.average_power(y.real) / sdr.average_power(y.imag))

    assert ratio_before == pytest.approx(0, abs=1e-12)
    assert ratio_after - ratio_before == pytest.approx(amplitude)


def test_phase():
    phase = 20  # degrees

    x = np.array([1 + 0j, 0 + 1j])
    y = sdr.iq_imbalance(x, 0, phase)

    angle_i = np.rad2deg(np.angle(y[0]))
    angle_q = np.rad2deg(np.angle(y[1]))

    assert angle_i == pytest.approx(-phase / 2)
    assert angle_q == pytest.approx(90 + phase / 2)
    assert angle_q - angle_i == pytest.approx(90 + phase)


def test_amplitude_and_phase():
    amplitude = 6
    phase = 30

    x = np.array([1 + 0j, 0 + 1j])
    y = sdr.iq_imbalance(x, amplitude, phase)

    gain_i = np.abs(y[0])
    gain_q = np.abs(y[1])

    assert 20 * np.log10(gain_i / gain_q) == pytest.approx(amplitude)
    assert np.rad2deg(np.angle(y[0])) == pytest.approx(-phase / 2)
    assert np.rad2deg(np.angle(y[1])) == pytest.approx(90 + phase / 2)
