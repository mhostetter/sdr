"""
References:
    - https://www.gaussianwaves.com/2020/09/equivalent-noise-bandwidth-enbw-of-window-functions/
"""

import pytest
import scipy.signal

import sdr


def test_boxcar():
    n = 128
    h = scipy.signal.windows.get_window("boxcar", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(1.000, rel=1e-3)


def test_barthann():
    n = 128
    h = scipy.signal.windows.get_window("barthann", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(1.456, rel=1e-3)


def test_bartlett():
    n = 128
    h = scipy.signal.windows.get_window("bartlett", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(1.333, rel=1e-3)


def test_blackman():
    n = 128
    h = scipy.signal.windows.get_window("blackman", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(1.727, rel=1e-3)


def test_blackmanharris():
    n = 128
    h = scipy.signal.windows.get_window("blackmanharris", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(2.004, rel=1e-3)


def test_bohman():
    n = 128
    h = scipy.signal.windows.get_window("bohman", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(1.786, rel=1e-3)


def test_flattop():
    n = 128
    h = scipy.signal.windows.get_window("flattop", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(3.770, rel=1e-3)


def test_hamming():
    n = 128
    h = scipy.signal.windows.get_window("hamming", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(1.363, rel=1e-3)


def test_hann():
    n = 128
    h = scipy.signal.windows.get_window("hann", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(1.500, rel=1e-3)


def test_nuttall():
    n = 128
    h = scipy.signal.windows.get_window("nuttall", n)
    fir = sdr.FIR(h)
    B_n = fir.noise_bandwidth(n)
    assert B_n == pytest.approx(1.976, rel=1e-3)
