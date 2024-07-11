import numpy as np

import sdr


def test_4():
    h = sdr.half_sine(4)
    h_truth = np.sqrt(2) / 2 * np.sin(np.pi * np.arange(4) / 4)
    np.testing.assert_array_almost_equal(h, h_truth)


def test_4_3():
    h = sdr.half_sine(4, span=3)
    h_truth = np.concatenate(
        (
            np.zeros(4),
            np.sqrt(2) / 2 * np.sin(np.pi * np.arange(4) / 4),
            np.zeros(4),
        )
    )
    np.testing.assert_array_almost_equal(h, h_truth)
