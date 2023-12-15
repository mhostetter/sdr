import numpy as np
import pytest

import sdr


def test_exceptions():
    with pytest.raises(TypeError):
        sdr.Differentiator(4.0)
    with pytest.raises(ValueError):
        sdr.Differentiator(0)
    with pytest.raises(ValueError):
        sdr.Differentiator(3)

    with pytest.raises(ValueError):
        sdr.Differentiator(10, window="bad_window")


def test_10_hamming():
    fir = sdr.Differentiator(10, window="hamming")
    h = fir.taps
    h_truth = np.array(
        [
            0.016,
            -0.04196305,
            0.13261739,
            -0.34107391,
            0.91214782,
            0.0,
            -0.91214782,
            0.34107391,
            -0.13261739,
            0.04196305,
            -0.016,
        ]
    )
    assert np.allclose(h, h_truth)


def test_20_kaiser():
    fir = sdr.Differentiator(20, window=("kaiser", 0.5))
    h = fir.taps
    h_truth = np.array(
        [
            -0.09403062,
            0.10572284,
            -0.1201978,
            0.13864545,
            -0.16304937,
            0.19698045,
            -0.247581,
            0.33151712,
            -0.49878825,
            0.99939384,
            0.0,
            -0.99939384,
            0.49878825,
            -0.33151712,
            0.247581,
            -0.19698045,
            0.16304937,
            -0.13864545,
            0.1201978,
            -0.10572284,
            0.09403062,
        ]
    )
    assert np.allclose(h, h_truth)


def test_central_difference():
    fir = sdr.Differentiator(2, window=None)
    h = fir.taps
    h_truth = np.array([1, 0, -1])
    assert np.allclose(h, h_truth)
