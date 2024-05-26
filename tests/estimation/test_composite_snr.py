import numpy as np

import sdr


def test_same():
    snr1 = np.arange(-40, 40 + 10, 10)
    snr = sdr.composite_snr(snr1, snr1)
    snr_truth = np.array(
        [
            -80.0008685,
            -60.00867722,
            -40.08600172,
            -20.79181246,
            -4.77121255,
            6.77780705,
            16.96803943,
            26.98752911,
            36.9894829,
        ]
    )
    assert np.allclose(snr, snr_truth)


def test_different_n20():
    snr1 = np.arange(-40, 40 + 10, 10)
    snr2 = -20
    snr = sdr.composite_snr(snr1, snr2)
    snr_truth = np.array(
        [
            -60.04364371,
            -50.04751156,
            -40.08600172,
            -30.45322979,
            -23.03196057,
            -20.41787319,
            -20.04364371,
            -20.00438416,
            -20.00043862,
        ]
    )
    assert np.allclose(snr, snr_truth)


def test_different_20():
    snr1 = np.arange(-40, 40 + 10, 10)
    snr2 = 20
    snr = sdr.composite_snr(snr1, snr2)
    snr_truth = np.array(
        [
            -40.04321804,
            -30.04325674,
            -20.04364371,
            -10.04751156,
            -0.08600172,
            9.54677021,
            16.96803943,
            19.58212681,
            19.95635629,
        ]
    )
    assert np.allclose(snr, snr_truth)
