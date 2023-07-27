import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.signal

import sdr


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_non_streaming(mode):
    N = 50
    x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal
    h = np.random.randn(10) + 1j * np.random.randn(10)  # FIR impulse response
    r = np.random.randint(3, 7)  # Interpolation rate

    fir = sdr.FIRInterpolator(h, r)
    y = fir.filter(x, mode)

    xr = np.zeros(N * r, dtype=np.complex64)
    xr[::r] = x[:]
    y_truth = scipy.signal.convolve(xr, h, mode=mode)

    if mode == "full":
        # Given the polyphase decomposition, the output is slightly less.
        y_truth = y_truth[0 : y.size]

    assert y.shape == y_truth.shape
    assert np.allclose(y, y_truth)


def test_streaming():
    N = 50
    x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal
    h = np.random.randn(10) + 1j * np.random.randn(10)  # FIR impulse response
    r = np.random.randint(3, 7)  # Interpolation rate

    fir = sdr.FIRInterpolator(h, r, streaming=True)

    d = 10  # Stride
    y = np.zeros(N * r, dtype=np.complex64)
    for i in range(0, N, d):
        y[i * r : (i + d) * r] = fir.filter(x[i : i + d])

    xr = np.zeros(N * r, dtype=np.complex64)
    xr[::r] = x[:]
    y_truth = scipy.signal.convolve(xr, h, mode="full")[0 : N * r]

    assert y.shape == y_truth.shape
    assert np.allclose(y, y_truth)


# def test_impulse_response():
#     h_truth = np.random.randn(10) + 1j * np.random.randn(10)  # FIR impulse response
#     r = np.random.randint(3, 7)  # Interpolation rate

#     fir = sdr.FIR(h_truth, r)

#     h = fir.impulse_response()
#     assert np.allclose(h, h_truth)

#     h = fir.impulse_response(20)
#     assert np.allclose(h, np.concatenate((h_truth, [0] * 10)))
