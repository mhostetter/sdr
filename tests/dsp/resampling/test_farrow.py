import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.signal

import sdr


def generate_signal():
    N = 1_000  # samples
    omega = 2 * np.pi / 40  # radians
    x = np.exp(1j * omega * np.arange(N))  # Complex exponential input signal
    x *= np.exp(-np.arange(N) / 300)  # Exponential decay
    tx = np.arange(N)  # Time axis for the input signal
    return tx, x


RATES = [1 / 2, 1 / 3, 1 / np.pi, 1 / 11, 2, 3, np.pi, 11]


@pytest.mark.parametrize("rate", RATES)
def test_match_scipy(rate):
    _, x = generate_signal()

    farrow = sdr.FarrowResampler()
    y = farrow(x, rate)
    y_scipy = scipy.signal.resample(x, int(math.ceil(rate * x.size)))

    # Ignore edge effects
    y = y[10:-10]
    y_scipy = y_scipy[10:-10]

    if False:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(y.real, label="y.real")
        plt.plot(y_scipy.real, label="y_scipy.real")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(y.imag, label="y.imag")
        plt.plot(y_scipy.imag, label="y_scipy.imag")
        plt.legend()
        plt.title(f"rate = {rate}")
        plt.show()

    np.testing.assert_array_almost_equal(y, y_scipy, decimal=1)


@pytest.mark.parametrize("rate", RATES)
def test_streaming_match_non_streaming(rate):
    _, x = generate_signal()

    farrow = sdr.FarrowResampler()
    y_non_streaming = farrow(x, rate)

    farrow = sdr.FarrowResampler(streaming=True)
    y_streaming = []
    N = 50
    assert x.size % N == 0
    for i in range(0, x.size, N):
        y_streaming.append(farrow(x[i : i + N], rate))
    y_streaming = np.concatenate(y_streaming)

    # Match the lengths
    y_non_streaming = y_non_streaming[0 : y_streaming.size]

    if False:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(y_non_streaming.real, label="y_non_streaming.real")
        plt.plot(y_streaming.real, label="y_streaming.real")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(y_non_streaming.imag, label="y_non_streaming.imag")
        plt.plot(y_streaming.imag, label="y_streaming.imag")
        plt.legend()
        plt.suptitle(f"rate = {rate}")
        plt.show()

    np.testing.assert_array_almost_equal(y_streaming, y_non_streaming)
