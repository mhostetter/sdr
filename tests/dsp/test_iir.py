import numpy as np
import pytest

import sdr


@pytest.mark.parametrize(
    "pole", [0.8, 0.8 * np.exp(1j * np.pi / 4), 1.2, 1.2 * np.exp(1j * np.pi / 4), 1.2, 1.2 * np.exp(1j * np.pi / 4)]
)
def test_single_pole_impulse_response(pole):
    """
    Reference:
        - R. Lyons, Understanding Digital Signal Processing 3rd Edition, Section 6.3.1.
    """
    N = 100
    x = np.zeros(N, dtype=float)
    x[0] = 1

    b = np.array([1], dtype=complex)
    a = np.array([1, -pole], dtype=complex)
    iir = sdr.IIR(b, a)
    h = iir(x)
    h_truth = pole ** np.arange(N)

    assert h.shape == h_truth.shape
    assert np.allclose(h, h_truth)


def test_streaming_match_full():
    N = 50
    x = np.random.randn(N) + 1j * np.random.randn(N)  # Input signal

    # Define random zeros and poles
    Nz = np.random.randint(1, 5)
    zeros = np.random.uniform(0.5, 1, size=Nz) * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=Nz))
    Np = np.random.randint(1, 5)
    poles = np.random.uniform(0.2, 0.8, size=Np) * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=Np))

    iir1 = sdr.IIR.ZerosPoles(zeros, poles)
    y_full = iir1(x)

    iir2 = sdr.IIR.ZerosPoles(zeros, poles, streaming=True)
    d = 10  # Stride
    y_stream = np.zeros_like(y_full)
    for i in range(0, N, d):
        y_stream[i : i + d] = iir2(x[i : i + d])

    np.testing.assert_array_almost_equal(y_full, y_stream)
