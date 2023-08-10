import numpy as np
import pytest
import scipy.signal

import sdr


@pytest.mark.parametrize("mode", ["full", "valid", "same"])
def test_non_streaming(mode):
    N = 50
    x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal
    h = np.random.randn(10) + 1j * np.random.randn(10)  # FIR impulse response

    fir = sdr.FIR(h)
    y = fir(x, mode)
    y_truth = scipy.signal.convolve(x, h, mode=mode)

    assert y.shape == y_truth.shape
    assert np.allclose(y, y_truth)


def test_streaming():
    N = 50
    x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal
    h = np.random.randn(10) + 1j * np.random.randn(10)  # FIR impulse response

    fir = sdr.FIR(h, streaming=True)

    d = 5  # Stride
    y = np.zeros(N, dtype=np.complex64)
    for i in range(0, N, d):
        y[i : i + d] = fir(x[i : i + d])

    y_truth = scipy.signal.convolve(x, h, mode="full")[0:N]

    assert y.shape == y_truth.shape
    assert np.allclose(y, y_truth)


def test_impulse_response():
    h_truth = np.random.randn(10) + 1j * np.random.randn(10)  # FIR impulse response
    fir = sdr.FIR(h_truth)

    h = fir.impulse_response()
    assert h.shape == h_truth.shape
    assert np.allclose(h, h_truth)

    h = fir.impulse_response(20)
    h_truth = np.concatenate((h_truth, [0] * 10))
    assert h.shape == h_truth.shape
    assert np.allclose(h, h_truth)
