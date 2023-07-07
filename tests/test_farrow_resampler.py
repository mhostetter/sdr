import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.signal

import sdr


def generate_signal():
    N = 100  # samples
    omega = 2 * np.pi / 20  # radians
    x = np.exp(1j * omega * np.arange(0, N))  # Complex exponential input signal
    x *= np.exp(-np.arange(N) / 100)  # Exponential decay
    tx = np.arange(0, N)  # Time axis for the input signal
    return tx, x


@pytest.mark.parametrize("rate", [1 / 2, 1 / 3, 1 / np.pi, 2, 3, np.pi])
def test_match_scipy(rate):
    _, x = generate_signal()
    rate = 2

    farrow = sdr.FarrowResampler()
    y = farrow.resample(x, rate)
    y_scipy = scipy.signal.resample(x, rate * x.size)

    # Remove filter delay and trim end to match signal lengths
    y = y[4 : 4 + y_scipy.size]

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
        plt.show()

    np.testing.assert_array_almost_equal(y, y_scipy, decimal=1)
