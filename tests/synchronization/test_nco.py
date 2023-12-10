import numpy as np
import pytest

import sdr


def test_call_exceptions():
    nco = sdr.NCO()
    with pytest.raises(ValueError):
        nco()
    with pytest.raises(ValueError):
        nco(np.array([1, 2, 3]), output="invalid")


def test_initial_phase_is_offset():
    gain = 1
    increment = 2 * np.pi / 10
    offset = np.pi / 2
    nco = sdr.NCO(gain, increment, offset)

    assert nco.gain == gain
    assert nco.increment == increment
    assert nco.offset == offset

    y = nco.step(1)
    y_truth = np.exp(1j * offset)
    assert np.allclose(y, y_truth)


def test_constant_increment():
    gain = 1
    increment = 0.1
    offset = 0
    nco = sdr.NCO(gain, increment, offset)

    assert nco.gain == gain
    assert nco.increment == increment
    assert nco.offset == offset

    N = 100
    y = nco.step(N)
    y_truth = np.exp(1j * (np.arange(N) * increment + offset))
    assert np.allclose(y, y_truth)


def test_constant_increment_constant_offset():
    gain = 1
    increment = 0.1
    offset = 0.5
    nco = sdr.NCO(gain, increment, offset)

    assert nco.gain == gain
    assert nco.increment == increment
    assert nco.offset == offset

    N = 100
    y = nco.step(N)
    y_truth = np.exp(1j * (np.arange(N) * increment + offset))
    assert np.allclose(y, y_truth)


def test_varying_phase():
    nco = sdr.NCO()

    phase = np.array([0, 1, 0, 1, 0, 1]) * np.pi
    phase = np.repeat(phase, 5)

    y = nco(phase=phase)
    y_truth = np.exp(1j * phase)
    assert np.allclose(y, y_truth)
