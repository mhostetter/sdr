import numpy as np

import sdr


def test_4():
    h = sdr.rectangular(4)
    h_truth = 0.5 * np.ones(4)
    assert np.array_equal(h, h_truth)


def test_4_3():
    h = sdr.rectangular(4, span=3)
    h_truth = np.concatenate(
        (
            np.zeros(4),
            0.5 * np.ones(4),
            np.zeros(4),
        )
    )
    assert np.array_equal(h, h_truth)
