import numpy as np
import pytest

import sdr


def test_psk_rect():
    rng = np.random.default_rng(0)
    psk = sdr.PSK(2, pulse_shape="rect")
    s = rng.integers(0, psk.order, 1_000)
    x = psk.modulate(s)
    b_rms = sdr.rms_bandwidth(x, sample_rate=psk.samples_per_symbol)
    assert b_rms == pytest.approx(0.7461700620944993, rel=1e-3)


def test_psk_srrc():
    rng = np.random.default_rng(0)
    psk = sdr.PSK(2, pulse_shape="srrc")
    s = rng.integers(0, psk.order, 1_000)
    x = psk.modulate(s)
    b_rms = sdr.rms_bandwidth(x, sample_rate=psk.samples_per_symbol)
    assert b_rms == pytest.approx(0.2900015177082325, rel=1e-3)
