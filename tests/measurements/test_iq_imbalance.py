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


# def test_phase():
#     # TODO: How to test phase imbalance?
#     return
