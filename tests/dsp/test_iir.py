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
    x = np.zeros(N, dtype=np.float32)
    x[0] = 1

    b = np.array([1], dtype=np.complex64)
    a = np.array([1, -pole], dtype=np.complex64)
    iir = sdr.IIR(b, a)
    h = iir(x)
    h_truth = pole ** np.arange(N)

    assert h.shape == h_truth.shape
    assert np.allclose(h, h_truth)
