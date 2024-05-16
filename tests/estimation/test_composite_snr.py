import numpy as np

import sdr


def test_same():
    snr1 = np.arange(-40, 40 + 10, 10)
    snr = sdr.composite_snr(snr1, snr1)
    snr_truth = np.array(
        [
            -76.99056855,
            -56.99837726,
            -37.07570176,
            -17.7815125,
            -1.76091259,
            9.78810701,
            19.97833938,
            29.99782907,
            39.99978286,
        ]
    )
    assert np.allclose(snr, snr_truth)


def test_different_n20():
    snr1 = np.arange(-40, 40 + 10, 10)
    snr2 = -20
    snr = sdr.composite_snr(snr1, snr2)
    snr_truth = np.array(
        [
            -57.03334375,
            -47.0372116,
            -37.07570176,
            -27.44292983,
            -20.02166062,
            -17.40757323,
            -17.03334375,
            -16.9940842,
            -16.99013866,
        ]
    )
    assert np.allclose(snr, snr_truth)


def test_different_20():
    snr1 = np.arange(-40, 40 + 10, 10)
    snr2 = 20
    snr = sdr.composite_snr(snr1, snr2)
    snr_truth = np.array(
        [
            -37.03291808,
            -27.03295678,
            -17.03334375,
            -7.0372116,
            2.92429824,
            12.55707017,
            19.97833938,
            22.59242677,
            22.96665625,
        ]
    )
    assert np.allclose(snr, snr_truth)
