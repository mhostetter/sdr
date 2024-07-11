import numpy as np
import pytest

import sdr


def test_psk_rect():
    rng = np.random.default_rng(0)
    psk = sdr.PSK(2, pulse_shape="rect", samples_per_symbol=100)
    s = rng.integers(0, psk.order, 1_000)  # Signal is 1000 seconds long
    x = psk.modulate(s)
    t_rms = sdr.rms_integration_time(x, sample_rate=psk.samples_per_symbol)
    assert t_rms == pytest.approx(288.672247843467, 1e-3)


def test_psk_srrc():
    rng = np.random.default_rng(0)
    psk = sdr.PSK(2, pulse_shape="srrc", samples_per_symbol=100)
    s = rng.integers(0, psk.order, 1_000)  # Signal is 1000 seconds long
    x = psk.modulate(s)
    t_rms = sdr.rms_integration_time(x, sample_rate=psk.samples_per_symbol)
    assert t_rms == pytest.approx(288.66026000704875, 1e-3)


def test_psk_srrc_parabolic():
    rng = np.random.default_rng(0)
    psk = sdr.PSK(2, pulse_shape="srrc", samples_per_symbol=100)
    s = rng.integers(0, psk.order, 1_000)  # Signal is 1000 seconds long
    x = psk.modulate(s)
    y = x * np.linspace(-1, 1, len(x)) ** 2  # Parabolic pulse shape
    y *= np.sqrt(sdr.energy(x) / sdr.energy(y))  # Normalize energy
    t_rms = sdr.rms_integration_time(y, sample_rate=psk.samples_per_symbol)
    assert t_rms == pytest.approx(422.61669635555705, 1e-3)
